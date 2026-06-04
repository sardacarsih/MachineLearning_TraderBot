import sys
import unittest
from dataclasses import fields
from types import SimpleNamespace

from config.settings import config
from config.mt5_config import MT5Config, mt5_config
from mt5.order_executor import OrderExecutor


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_FOK = 2
    ORDER_FILLING_RETURN = 3
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE_PARTIAL = 10010
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def __init__(
        self,
        margin_free=1000.0,
        margin_per_lot=100.0,
        success_retcode=None,
        success_comment="",
        positions=None,
        bid=2000.0,
        ask=2000.5,
    ):
        self.margin_free = margin_free
        self.margin_per_lot = margin_per_lot
        self.success_retcode = success_retcode
        self.success_comment = success_comment
        self.positions = positions or []
        self.bid = bid
        self.ask = ask
        self.sent_requests = []

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=self.bid, ask=self.ask)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            filling_mode=2,
            point=0.01,
        )

    def account_info(self):
        return SimpleNamespace(margin_free=self.margin_free)

    def order_calc_margin(self, action_type, symbol, lot, price):
        return lot * self.margin_per_lot

    def order_check(self, request):
        if request["volume"] * self.margin_per_lot <= self.margin_free * (1.0 - mt5_config.margin_reserve_pct):
            retcode = self.TRADE_RETCODE_DONE if self.success_retcode is None else self.success_retcode
            return SimpleNamespace(retcode=retcode, comment=self.success_comment)
        return SimpleNamespace(retcode=10019)

    def order_send(self, request):
        self.sent_requests.append(dict(request))
        return SimpleNamespace(
            retcode=self.TRADE_RETCODE_DONE,
            order=123,
            volume=request["volume"],
            price=request["price"],
            request=SimpleNamespace(sl=request["sl"], tp=request["tp"]),
        )

    def positions_get(self, symbol=None, ticket=None):
        if ticket is not None:
            return [pos for pos in self.positions if getattr(pos, "ticket", None) == ticket]
        if symbol is not None:
            return [pos for pos in self.positions if getattr(pos, "symbol", None) == symbol]
        return list(self.positions)

    def last_error(self):
        return (0, "OK")


class FakeConnector:
    def __init__(self, mt5):
        self._mt5 = mt5

    def is_connected(self):
        return True

    def get_symbol_info(self, symbol):
        return self._mt5.symbol_info(symbol)


class OrderSafetyTests(unittest.TestCase):
    def setUp(self):
        self._original_module = sys.modules.get("MetaTrader5")
        self._mt5_original = {cfg_field.name: getattr(mt5_config, cfg_field.name) for cfg_field in fields(MT5Config)}
        mt5_config.paper_trading = False
        mt5_config.trading_enabled = True
        mt5_config.safety_max_lot = 1.0
        mt5_config.margin_reserve_pct = 0.10
        self._max_open_positions = config.risk.max_open_positions
        config.risk.max_open_positions = 3

    def tearDown(self):
        if self._original_module is None:
            sys.modules.pop("MetaTrader5", None)
        else:
            sys.modules["MetaTrader5"] = self._original_module
        for name, value in self._mt5_original.items():
            setattr(mt5_config, name, value)
        config.risk.max_open_positions = self._max_open_positions

    def make_position(self, symbol="XAUUSD", pos_type=None, volume=0.1, price_open=2000.0, magic=999, ticket=1):
        if pos_type is None:
            pos_type = FakeMT5.POSITION_TYPE_BUY
        return SimpleNamespace(
            ticket=ticket,
            symbol=symbol,
            type=pos_type,
            volume=volume,
            price_open=price_open,
            magic=magic,
        )

    def test_oversized_lot_is_downsized_to_margin_safe_volume(self):
        fake_mt5 = FakeMT5(margin_free=50.0, margin_per_lot=100.0)
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, ticket = executor.open_position("XAUUSD", "BUY", 100.0, 1990.0, 2020.0)

        self.assertTrue(success)
        self.assertEqual(ticket, 123)
        self.assertEqual(fake_mt5.sent_requests[0]["volume"], 0.45)

    def test_trade_is_skipped_when_min_lot_fails_margin(self):
        fake_mt5 = FakeMT5(margin_free=0.5, margin_per_lot=100.0)
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, reason = executor.open_position("XAUUSD", "BUY", 1.0, 1990.0, 2020.0)

        self.assertFalse(success)
        self.assertIn("Trade skipped", reason)
        self.assertEqual(fake_mt5.sent_requests, [])

    def test_order_check_retcode_zero_done_is_accepted(self):
        fake_mt5 = FakeMT5(margin_free=10_000_000.0, margin_per_lot=136_860_286.0, success_retcode=0, success_comment="Done")
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, ticket = executor.open_position("USTEC_x100", "SELL", 0.01, 30680.0, 30580.0)

        self.assertTrue(success)
        self.assertEqual(ticket, 123)
        self.assertEqual(fake_mt5.sent_requests[0]["volume"], 0.01)

    def test_first_symbol_position_is_allowed(self):
        fake_mt5 = FakeMT5()
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, ticket = executor.open_position("XAUUSD", "BUY", 0.1, 1990.0, 2020.0)

        self.assertTrue(success)
        self.assertEqual(ticket, 123)
        self.assertEqual(len(fake_mt5.sent_requests), 1)

    def test_fourth_symbol_position_is_blocked_across_magic_numbers(self):
        positions = [
            self.make_position(price_open=2010.0, magic=111, ticket=1),
            self.make_position(price_open=2005.0, magic=222, ticket=2),
            self.make_position(price_open=2001.0, magic=333, ticket=3),
        ]
        fake_mt5 = FakeMT5(positions=positions, bid=1999.0, ask=1999.5)
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, reason = executor.open_position("XAUUSD", "BUY", 0.1, 1990.0, 2020.0)

        self.assertFalse(success)
        self.assertIn("Max open positions", reason)
        self.assertEqual(fake_mt5.sent_requests, [])

    def test_opposite_direction_entry_is_blocked(self):
        positions = [self.make_position(pos_type=FakeMT5.POSITION_TYPE_SELL, price_open=2000.0)]
        fake_mt5 = FakeMT5(positions=positions)
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, reason = executor.open_position("XAUUSD", "BUY", 0.1, 1990.0, 2020.0)

        self.assertFalse(success)
        self.assertIn("existing direction is SELL", reason)
        self.assertEqual(fake_mt5.sent_requests, [])

    def test_buy_averaging_is_allowed_below_weighted_average_entry(self):
        positions = [
            self.make_position(volume=0.1, price_open=2010.0, ticket=1),
            self.make_position(volume=0.3, price_open=2000.0, ticket=2),
        ]
        fake_mt5 = FakeMT5(positions=positions, bid=1996.0, ask=1996.5)
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, ticket = executor.open_position("XAUUSD", "BUY", 0.1, 1990.0, 2020.0)

        self.assertTrue(success)
        self.assertEqual(ticket, 123)
        self.assertEqual(len(fake_mt5.sent_requests), 1)

    def test_sell_averaging_is_allowed_above_weighted_average_entry(self):
        positions = [
            self.make_position(pos_type=FakeMT5.POSITION_TYPE_SELL, volume=0.2, price_open=1990.0, ticket=1),
            self.make_position(pos_type=FakeMT5.POSITION_TYPE_SELL, volume=0.2, price_open=2000.0, ticket=2),
        ]
        fake_mt5 = FakeMT5(positions=positions, bid=2001.0, ask=2001.5)
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, ticket = executor.open_position("XAUUSD", "SELL", 0.1, 2010.0, 1980.0)

        self.assertTrue(success)
        self.assertEqual(ticket, 123)
        self.assertEqual(len(fake_mt5.sent_requests), 1)

    def test_same_direction_entry_is_blocked_when_price_is_not_worse(self):
        positions = [self.make_position(price_open=2000.5)]
        fake_mt5 = FakeMT5(positions=positions, bid=2000.0, ask=2000.5)
        sys.modules["MetaTrader5"] = fake_mt5
        executor = OrderExecutor(FakeConnector(fake_mt5))

        success, reason = executor.open_position("XAUUSD", "BUY", 0.1, 1990.0, 2020.0)

        self.assertFalse(success)
        self.assertIn("is not worse than average", reason)
        self.assertEqual(fake_mt5.sent_requests, [])


if __name__ == "__main__":
    unittest.main()
