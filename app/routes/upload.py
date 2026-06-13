import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
from flask_login import login_required
from werkzeug.utils import secure_filename
from app import logger

upload_bp = Blueprint('upload', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@upload_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        lane_files = {}
        errors = []

        for i in range(1, 5):
            key = f'lane{i}'
            file = request.files.get(key)
            if not file or file.filename == '':
                errors.append(f'Lane {i} video is required.')
            elif not allowed_file(file.filename):
                errors.append(f'Lane {i}: only .mp4, .avi, .mov, .mkv files allowed.')
            else:
                lane_files[key] = file

        if errors:
            for err in errors:
                flash(err, 'danger')
            return render_template('upload.html')

        saved_paths = {}
        upload_folder = current_app.config['UPLOAD_FOLDER']

        for key, file in lane_files.items():
            filename = secure_filename(f"{key}_{file.filename}")
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            saved_paths[key] = filepath
            logger.info(f'Saved {key} video: {filepath}')

        session['lane_videos'] = saved_paths
        flash('Videos uploaded successfully. Starting analysis...', 'success')
        return redirect(url_for('analysis.run_analysis'))

    return render_template('upload.html')
