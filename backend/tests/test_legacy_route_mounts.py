from backend.app import app


def test_legacy_routes_remain_mounted():
    mounted_paths = {route.path for route in app.routes}

    assert "/api/health" in mounted_paths
    assert "/api/llm_status" in mounted_paths


def test_legacy_routes_are_not_duplicated():
    route_counts = {}
    for route in app.routes:
        methods = frozenset((route.methods or set()) - {"HEAD", "OPTIONS"})
        key = (route.path, methods)
        route_counts[key] = route_counts.get(key, 0) + 1

    singleton_routes = [
        ("/api/health", frozenset({"GET"})),
        ("/api/llm_status", frozenset({"GET"})),
    ]
    for route_key in singleton_routes:
        assert route_counts.get(route_key) == 1
