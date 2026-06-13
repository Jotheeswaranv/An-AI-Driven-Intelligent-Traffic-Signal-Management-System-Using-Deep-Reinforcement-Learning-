from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import check_password_hash
from app.models.user import User
from app import logger

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/', methods=['GET', 'POST'])
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            logger.info(f'User "{username}" logged in.')
            return redirect(url_for('dashboard.index'))

        flash('Invalid username or password.', 'danger')
        logger.warning(f'Failed login attempt for username: "{username}"')

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logger.info('User logged out.')
    logout_user()
    return redirect(url_for('auth.login'))
