from fastapi import Request
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
from collections import defaultdict, deque
import time
import psutil
import asyncio

# Create registry
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

REQUEST_SIZE = Histogram(
    'api_request_size_bytes',
    'Request size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

RESPONSE_SIZE = Histogram(
    'api_response_size_bytes',
    'Response size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Number of active requests',
    ['method', 'endpoint'],
    registry=registry
)

ERROR_COUNT = Counter(
    'api_errors_total',
    'Total API errors',
    ['method', 'endpoint', 'status'],
    registry=registry
)

# In-memory storage for detailed analytics
analytics_data = defaultdict(lambda: {
    'total_requests': 0,
    'success_count': 0,
    'error_count': 0,
    'total_latency': 0,
    'latencies': deque(maxlen=1000),  # Store last 1000 latencies
    'request_sizes': deque(maxlen=1000),
    'response_sizes': deque(maxlen=1000),
    'status_codes': defaultdict(int),
    'last_request': None,
    'avg_latency': 0,
    'p95_latency': 0,
    'p99_latency': 0,
    'min_latency': float('inf'),
    'max_latency': 0,
})

# System metrics
system_metrics = {
    'cpu_percent': 0,
    'memory_percent': 0,
    'disk_usage': 0,
    'network_connections': 0
}

class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip monitoring routes
        if request.url.path in ['/metrics', '/admin/analytics']:
            return await call_next(request)
        
        method = request.method
        endpoint = request.url.path
        key = f"{method}:{endpoint}"
        
        # Track active requests
        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
        
        # Start timer
        start_time = time.time()
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        
        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Get response size
            response_size = int(response.headers.get('content-length', 0))
            
            # Update Prometheus metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(latency)
            
            REQUEST_SIZE.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
            
            RESPONSE_SIZE.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_size)
            
            # Update analytics data
            data = analytics_data[key]
            data['total_requests'] += 1
            data['latencies'].append(latency * 1000)  # Convert to ms
            data['request_sizes'].append(request_size)
            data['response_sizes'].append(response_size)
            data['status_codes'][status_code] += 1
            data['last_request'] = datetime.utcnow().isoformat()
            
            if 200 <= status_code < 400:
                data['success_count'] += 1
            else:
                data['error_count'] += 1
                ERROR_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status_code
                ).inc()
            
            # Calculate statistics
            if data['latencies']:
                latencies_list = list(data['latencies'])
                latencies_sorted = sorted(latencies_list)
                data['avg_latency'] = sum(latencies_list) / len(latencies_list)
                data['min_latency'] = min(latencies_list)
                data['max_latency'] = max(latencies_list)
                data['p95_latency'] = latencies_sorted[int(len(latencies_sorted) * 0.95)] if len(latencies_sorted) > 0 else 0
                data['p99_latency'] = latencies_sorted[int(len(latencies_sorted) * 0.99)] if len(latencies_sorted) > 0 else 0
            
            return response
            
        except Exception as e:
            ERROR_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=500
            ).inc()
            raise
        finally:
            ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()

# System metrics collector
async def update_system_metrics():
    while True:
        system_metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        system_metrics['memory_percent'] = psutil.virtual_memory().percent
        system_metrics['disk_usage'] = psutil.disk_usage('/').percent
        system_metrics['network_connections'] = len(psutil.net_connections())
        await asyncio.sleep(5)  # Update every 5 seconds

def get_analytics_summary():
    """Get comprehensive analytics summary"""
    summary = []
    
    for key, data in analytics_data.items():
        method, endpoint = key.split(':', 1)
        
        # Calculate error rate
        error_rate = (data['error_count'] / data['total_requests'] * 100) if data['total_requests'] > 0 else 0
        
        # Calculate average sizes
        avg_request_size = sum(data['request_sizes']) / len(data['request_sizes']) if data['request_sizes'] else 0
        avg_response_size = sum(data['response_sizes']) / len(data['response_sizes']) if data['response_sizes'] else 0
        
        summary.append({
            'method': method,
            'endpoint': endpoint,
            'total_requests': data['total_requests'],
            'success_count': data['success_count'],
            'error_count': data['error_count'],
            'error_rate': round(error_rate, 2),
            'avg_latency_ms': round(data['avg_latency'], 2),
            'min_latency_ms': round(data['min_latency'], 2),
            'max_latency_ms': round(data['max_latency'], 2),
            'p95_latency_ms': round(data['p95_latency'], 2),
            'p99_latency_ms': round(data['p99_latency'], 2),
            'avg_request_size_bytes': round(avg_request_size, 2),
            'avg_response_size_bytes': round(avg_response_size, 2),
            'status_codes': dict(data['status_codes']),
            'last_request': data['last_request']
        })
    
    return sorted(summary, key=lambda x: x['total_requests'], reverse=True)

# Create Prometheus metrics app
metrics_app = make_asgi_app(registry=registry)
