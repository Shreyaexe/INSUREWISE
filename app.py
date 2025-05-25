from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from chatbot.chatbot import get_chatbot_response
from werkzeug.security import generate_password_hash, check_password_hash
import markdown
import re
import sqlite3
import time

app = Flask(__name__)
app.secret_key = "insurewise123"

DB_NAME = 'users.db'

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL
                    )''')
        conn.commit()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
            user = c.fetchone()

            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                flash("Logged in successfully!")
                return redirect(url_for('index'))
            else:
                flash("Invalid email or password.")
                return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Email validation
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
            flash("Invalid email format. Please enter a valid email address.")
            return render_template('signup.html', name=name, email=email)

        # Password validation
        if password != confirm_password:
            flash("Passwords don't match. Please make sure both password fields are the same.")
            return render_template('signup.html', name=name, email=email)

        # Password strength check
        if not re.match(r'^[A-Z][A-Za-z0-9]*[0-9]+.*$', password):
            flash("Password must start with an uppercase letter and contain at least one number.")
            return render_template('signup.html', name=name, email=email)

        try:
            hashed_password = generate_password_hash(password)
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                          (name, email, hashed_password))
                conn.commit()
                flash("Account created successfully! Please log in.")
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists. Try logging in.")
            return render_template('signup.html', name=name, email=email)

    return render_template('signup.html')



@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user_id' not in session:
        if request.method == 'POST':
            return jsonify({'error': 'Please log in to use the chatbot.'}), 401
        flash("Please log in to use the chatbot.") 
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_message = request.form.get('user_message')
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
            
        try:
            bot_reply = get_chatbot_response(user_message)
            # Return the response as plain text
            return bot_reply
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return jsonify({'error': 'Failed to generate response'}), 500

    # For GET requests, show the chat page with any existing messages
    return render_template('chat.html')

if __name__ == '__main__': 
    app.run(debug=True, host="0.0.0.0", port=4000)