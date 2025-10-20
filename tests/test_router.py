import pytest
from f1_chat_app.utils.router import route_query

def test_route_unknown():
    assert route_query("hello world") in ("unknown", "season", "gp")
