import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = os.environ.get('BOT_TOKEN')

# OTC-пары Pocket Option
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X"
}

TIMEFRAMES = {
    "1m": "1m",
    "3m": "5m",
    "5m": "5m"
}

logging.basicConfig(level=logging.INFO)

model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)

def get_data(symbol, interval, period="7d"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return pd.DataFrame()

def prepare_features(df):
    if len(df) < 30:
        return None
    
    close = df['Close']
    rsi = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    ema_fast = EMAIndicator(close=close, window=9)
    ema_slow = EMAIndicator(close=close, window=21)
    
    features = pd.DataFrame({
        'rsi': rsi,
        'macd': macd.macd(),
        'macd_signal': macd.macd_signal(),
        'ema_fast': ema_fast.ema_indicator(),
        'ema_slow': ema_slow.ema_indicator(),
        'close': close,
        'volume': df['Volume'],
        'high_low': df['High'] / df['Low'],
    })
    
    for i in range(1, 4):
        features[f'close_lag_{i}'] = close.shift(i)
        features[f'rsi_lag_{i}'] = rsi.shift(i)
    
    return features.dropna()

def train_model(df):
    features = prepare_features(df)
    if features is None or len(features) < 30:
        return False
    
    future_close = df['Close'].shift(-5)
    target = (future_close > df['Close']).astype(int)
    
    X = features.iloc[:-5]
    y = target.iloc[:-5]
    
    if len(X) > 20:
        model.fit(X, y)
        return True
    return False

def generate_signal(df):
    if df.empty or len(df) < 30:
        return {"signal": "WAIT", "confidence": 0, "reason": "Недостаточно данных"}
    
    features = prepare_features(df)
    if features is None:
        return {"signal": "WAIT", "confidence": 0, "reason": "Нет признаков"}
    
    last_features = features.iloc[-1:].values
    
    try:
        prob = model.predict_proba(last_features)[0]
        pred = model.predict(last_features)[0]
        confidence = max(prob) * 100
    except:
        rsi = RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1]
        macd = MACD(close=df['Close'])
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        
        score = 0
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2
        if macd_line > macd_signal:
            score += 1.5
        else:
            score -= 1.5
        
        confidence = min(abs(score) * 20, 75)
        pred = 1 if score > 0 else 0
    
    signal = "CALL" if pred == 1 and confidence > 55 else "PUT" if pred == 0 and confidence > 55 else "WAIT"
    
    return {
        "signal": signal,
        "confidence": round(confidence, 1),
        "reason": "Нейросеть" if confidence > 55 else "Недостаточно уверенности",
        "price": round(df['Close'].iloc[-1], 4),
        "rsi": round(RSIIndicator(close=df['Close'], window=14).rsi().iloc[-1], 1)
    }

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    for pair in PAIRS.keys():
        keyboard.append([InlineKeyboardButton(pair, callback_data=f"pair_{pair}")])
    
    await update.message.reply_text(
        "🤖 **Торговый бот с нейросетью**\n\n"
        "Выберите пару:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("pair_"):
        pair = data.replace("pair_", "")
        context.user_data['pair'] = pair
        context.user_data['symbol'] = PAIRS[pair]
        
        df = get_data(PAIRS[pair], interval="1m", period="7d")
        if not df.empty:
            train_model(df)
            context.user_data['trained'] = True
        
        keyboard = []
        for tf in TIMEFRAMES.keys():
            keyboard.append([InlineKeyboardButton(tf, callback_data=f"tf_{tf}")])
        
        await query.edit_message_text(
            f"📊 **{pair}**\n"
            "🧠 Нейросеть обучена\n\n"
            "Выберите таймфрейм:",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data.startswith("tf_") or data.startswith("refresh_"):
        tf = data.replace("tf_", "").replace("refresh_", "")
        pair = context.user_data.get('pair', 'EUR/USD')
        symbol = context.user_data.get('symbol', 'EURUSD=X')
        yf_interval = TIMEFRAMES.get(tf, "1m")
        
        df = get_data(symbol, interval=yf_interval, period="1d")
        if df.empty:
            await query.edit_message_text("❌ Нет данных. Попробуйте позже.")
            return
        
        if context.user_data.get('trained', False):
            train_model(df)
        
        signal_data = generate_signal(df)
        
        emoji = "🟢" if signal_data['signal'] == 'CALL' else "🔴" if signal_data['signal'] == 'PUT' else "⚪"
        
        text = f"""
📈 **{pair}** | {tf}

{emoji} **Сигнал: {signal_data['signal']}**
🎯 Уверенность: {signal_data['confidence']}%

💰 Цена: ${signal_data['price']}
📊 RSI: {signal_data['rsi']}
🧠 {signal_data['reason']}

🕐 {datetime.now().strftime('%H:%M:%S')}
        """
        
        keyboard = [[InlineKeyboardButton("🔄 Обновить", callback_data=f"refresh_{tf}")]]
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    
    print("🤖 Бот с нейросетью запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()
