from backend.app import app


def _concrete_routes():
    """Flatten FastAPI's lazy included-router wrapper across supported versions."""
    for route in app.routes:
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            yield from original_router.routes
        elif hasattr(route, "path"):
            yield route


def test_legacy_routes_remain_mounted():
    mounted_paths = {route.path for route in _concrete_routes()}
    assert "/api/health" in mounted_paths
    assert "/api/llm_status" in mounted_paths


def test_legacy_routes_are_not_duplicated():
    route_counts = {}
    for route in _concrete_routes():
        methods = frozenset((route.methods or set()) - {"HEAD", "OPTIONS"})
        key = (route.path, methods)
        route_counts[key] = route_counts.get(key, 0) + 1

    singleton_routes = [
        ("/api/health", frozenset({"GET"})),
        ("/api/llm_status", frozenset({"GET"})),
    ]
    for route_key in singleton_routes:
        assert route_counts.get(route_key) == 1
