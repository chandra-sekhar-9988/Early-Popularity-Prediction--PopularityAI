"""
app.py
------
Flask web server — PopularityAI: Early Popularity Prediction
  Uses: popularity_model.pkl (HistGradientBoostingRegressor, 39 features)
  Trained by: train_model.py
  Predictor: src/predictor.py (PopularityPredictor)
"""

import os
import sys
import json
import pickle
import threading
import subprocess

from flask import Flask, request, jsonify, render_template, Response

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

# ── Model singleton ──────────────────────────────────────────────────────────
_predictor = None
_predictor_lock = threading.Lock()

def get_predictor():
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                from src.predictor import PopularityPredictor
                _predictor = PopularityPredictor()
    return _predictor


# ── Training state ───────────────────────────────────────────────────────────
_train_log  = []
_train_done = False
_train_lock = threading.Lock()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    trained = os.path.exists('saved/popularity_model.pkl')
    meta = {}
    if trained:
        try:
            with open('saved/popularity_model.pkl', 'rb') as f:
                bundle = pickle.load(f)
            meta = bundle.get('meta', {})
        except Exception:
            pass
    return jsonify({'model_trained': trained, 'meta': meta})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True) or {}
        predictor = get_predictor()
        result = predictor.predict(data)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({'error': 'Model not trained', 'message':
            'popularity_model.pkl not found. Click Train to build the model.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def api_train():
    """Fire-and-forget background training."""
    global _train_log, _train_done, _predictor
    with _train_lock:
        _train_log = []
        _train_done = False
        _predictor = None   # reset so next predict reloads

    def _run():
        global _train_done, _predictor
        try:
            proc = subprocess.Popen(
                [sys.executable, '-u', 'train_model.py'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace'
            )
            for line in proc.stdout:
                line = line.rstrip()
                with _train_lock:
                    _train_log.append(line)
            proc.wait()
            if proc.returncode == 0:
                _predictor = None  # force reload
                with _train_lock:
                    _train_log.append('__DONE__')
            else:
                with _train_lock:
                    _train_log.append(f'__ERROR__Training failed (exit {proc.returncode})')
        except Exception as ex:
            with _train_lock:
                _train_log.append(f'__ERROR__{ex}')
        _train_done = True

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'status': 'training started'})


@app.route('/api/train/stream')
def api_train_stream():
    """SSE endpoint — streams training log lines to the browser."""
    def generate():
        sent = 0
        while True:
            with _train_lock:
                lines = _train_log[sent:]
            for line in lines:
                yield f'data: {line}\n\n'
                sent += 1
                if line in ('__DONE__',) or line.startswith('__ERROR__'):
                    return
            if _train_done and sent >= len(_train_log):
                break
            import time; time.sleep(0.3)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Startup ──────────────────────────────────────────────────────────────────

def _warmup():
    """Pre-load model at startup for instant first prediction."""
    if os.path.exists('saved/popularity_model.pkl'):
        try:
            get_predictor()
            with open('saved/popularity_model.pkl', 'rb') as f:
                bundle = pickle.load(f)
            meta = bundle.get('meta', {})
            r2  = meta.get('r2', 0)
            w2x = meta.get('within2x', 0)
            print(f'  [OK] Popularity model ready — R²={r2:.4f}, Within-2x={w2x:.1f}%')
        except Exception as e:
            print(f'  [WARN] Model warm-up failed: {e}')
    else:
        print('  [WARN] No trained model found — click Train in the UI to build one')


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  PopularityAI — Early Popularity Prediction')
    print('  HistGradientBoosting + Graph Features')
    print('=' * 60)
    _warmup()
    print('\n  Open: http://127.0.0.1:5000\n')
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
