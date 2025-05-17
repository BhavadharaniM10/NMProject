# Restart each worker every 20 requests (tolerates small leaks)
gunicorn myapp.wsgi:application \
        --workers 2 \
        --worker-class gthread \
        --threads 4 \
        --timeout 90 \
        --max-requests 20 --max-requests-jitter 5
