import pytest
from backend.domain.models import BrokerStatus
from backend.services.runtime_state import market_data_dependency_state
from backend.adapters.ctrader import adapter
from backend import ctrader_client as ctd

def test_broker_status_structure():
    """Verify that BrokerStatus has the new granular fields."""
    status = adapter.get_status()
    assert hasattr(status, "socket_connected")
    assert hasattr(status, "account_authorized")
    assert hasattr(status, "market_data_ready")

def test_status_transitions():
    """Simulate state transitions and verify adapter reflects them."""
    # Reset state
    ctd.CONNECTED = False
    ctd.AUTHORIZED = False
    market_data_dependency_state.market_data_ready = False
    
    # State 1: Disconnected
    status = adapter.get_status()
    assert status.socket_connected is False
    assert status.account_authorized is False
    assert status.ready is False
    
    # State 2: Socket Connected, Auth Pending
    ctd.CONNECTED = True
    status = adapter.get_status()
    assert status.socket_connected is True
    assert status.account_authorized is False
    assert status.ready is False
    
    # State 3: Authorized
    ctd.AUTHORIZED = True
    # Note: ready also requires symbols_loaded > 0.
    # We'll assume symbols are loaded for this test to check logic.
    # adapter.get_status calls symbols_loaded = len(ctd.symbol_name_to_id or {})
    ctd.symbol_name_to_id = {"XAUUSD": 1} 
    status = adapter.get_status()
    assert status.socket_connected is True
    assert status.account_authorized is True
    assert status.ready is True
    
    # State 4: Market Data Ready
    market_data_dependency_state.market_data_ready = True
    status = adapter.get_status()
    assert status.market_data_ready is True
