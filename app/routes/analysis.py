import json
import threading
from flask import Blueprint, render_template, session, redirect, url_for, Response, stream_with_context
from flask_login import login_required
from app.services.video_processor import VideoProcessor
from app import logger

analysis_bp = Blueprint('analysis', __name__)

# Shared state for all lane results
analysis_state = {
    'results': {},
    'running': False,
    'done': False
}
state_lock = threading.Lock()


@analysis_bp.route('/analysis')
@login_required
def run_analysis():
    lane_videos = session.get('lane_videos')
    if not lane_videos:
        return redirect(url_for('upload.upload'))

    with state_lock:
        analysis_state['results'] = {}
        analysis_state['running'] = True
        analysis_state['done'] = False

    processor = VideoProcessor(lane_videos, analysis_state, state_lock)
    thread = threading.Thread(target=processor.process_all, daemon=True)
    thread.start()
    logger.info('Analysis thread started.')

    return render_template('analysis.html', lanes=list(lane_videos.keys()))


@analysis_bp.route('/analysis/stream')
@login_required
def stream():
    def event_generator():
        import time
        while True:
            with state_lock:
                data = {
                    'results': analysis_state['results'],
                    'done': analysis_state['done']
                }
            yield f"data: {json.dumps(data)}\n\n"
            if analysis_state['done']:
                break
            time.sleep(0.5)

    return Response(
        stream_with_context(event_generator()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )
