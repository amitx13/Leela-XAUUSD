"""
engines/starvation_tracker.py — Silent Trade Starvation Detection.

Tracks signal flow through the gate stack. Surfaces warnings when the system
is generating signals but executing none. Resets per session.

Phase 1A-2 component.
"""
from datetime import datetime, date
from utils.logger import log_event, log_warning
from db.connection import execute_write


class StarvationTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counters = {
            'signals_evaluated': 0,
            'signals_generated': 0,
            'blocked_regime': 0,
            'blocked_ks': 0,
            'blocked_spread': 0,
            'blocked_volume': 0,
            'blocked_adx_filter': 0,
            'blocked_compound_gate': 0,
            'blocked_family': 0,
            'blocked_portfolio': 0,
            'blocked_event': 0,
            'blocked_other': 0,
            'orders_placed': 0,
            'orders_filled': 0,
        }
        self.blocked_details = []

    def record_evaluation(self):
        self.counters['signals_evaluated'] += 1

    def record_signal(self):
        self.counters['signals_generated'] += 1

    def record_block(self, strategy: str, gate: str, context: str = ""):
        key = f'blocked_{gate}'
        if key in self.counters:
            self.counters[key] += 1
        else:
            self.counters['blocked_other'] += 1
        self.blocked_details.append({
            'time': datetime.utcnow().isoformat(),
            'strategy': strategy,
            'gate': gate,
            'context': context
        })

    def record_order(self):
        self.counters['orders_placed'] += 1

    def record_fill(self):
        self.counters['orders_filled'] += 1

    def check_starvation(self, session_name: str) -> list[str]:
        gen = self.counters['signals_generated']
        exe = self.counters['orders_placed']
        alerts = []

        if gen >= 3 and exe == 0:
            top_blocker = max(
                [(k, v) for k, v in self.counters.items()
                 if k.startswith('blocked_') and v > 0],
                key=lambda x: x[1],
                default=('none', 0)
            )
            alerts.append(
                f"STARVATION [{session_name}]: {gen} signals generated, "
                f"0 executed. Top blocker: {top_blocker[0]} ({top_blocker[1]}x)"
            )

        if self.counters['signals_evaluated'] > 0 and gen == 0:
            alerts.append(
                f"SILENT [{session_name}]: {self.counters['signals_evaluated']} "
                f"strategies evaluated, 0 signals generated."
            )

        if gen > 5 and exe < gen * 0.2:
            block_rate = 1 - (exe / gen) if gen > 0 else 0
            alerts.append(
                f"HIGH BLOCK RATE [{session_name}]: {block_rate:.0%} of signals blocked."
            )

        for alert in alerts:
            log_warning("STARVATION_ALERT", message=alert, session=session_name)

        return alerts

    def daily_summary(self) -> dict:
        import json
        summary = {
            'date': date.today().isoformat(),
            'total_evaluated': self.counters['signals_evaluated'],
            'total_generated': self.counters['signals_generated'],
            'total_executed': self.counters['orders_placed'],
            'total_filled': self.counters['orders_filled'],
            'pass_through_rate': (
                self.counters['orders_placed'] / self.counters['signals_generated']
                if self.counters['signals_generated'] > 0 else 0
            ),
            'top_blockers': sorted(
                [(k, v) for k, v in self.counters.items()
                 if k.startswith('blocked_') and v > 0],
                key=lambda x: x[1], reverse=True
            )[:3],
        }

        # Persist to DB
        top = summary['top_blockers']
        try:
            execute_write(
                """INSERT INTO system_state.starvation_log
                   (log_date, signals_evaluated, signals_generated,
                    orders_placed, orders_filled, pass_through_rate,
                    top_blocker_1, top_blocker_1_count,
                    top_blocker_2, top_blocker_2_count,
                    top_blocker_3, top_blocker_3_count,
                    details)
                   VALUES (:log_date, :eval, :gen, :placed, :filled, :ptr,
                           :tb1, :tb1c, :tb2, :tb2c, :tb3, :tb3c,
                           :details)""",
                {
                    'log_date': date.today(),
                    'eval': summary['total_evaluated'],
                    'gen': summary['total_generated'],
                    'placed': summary['total_executed'],
                    'filled': summary['total_filled'],
                    'ptr': round(summary['pass_through_rate'], 4),
                    'tb1': top[0][0] if len(top) > 0 else None,
                    'tb1c': top[0][1] if len(top) > 0 else None,
                    'tb2': top[1][0] if len(top) > 1 else None,
                    'tb2c': top[1][1] if len(top) > 1 else None,
                    'tb3': top[2][0] if len(top) > 2 else None,
                    'tb3c': top[2][1] if len(top) > 2 else None,
                    'details': json.dumps(self.blocked_details[-50:]),  # last 50 blocks
                }
            )
        except Exception as e:
            log_warning("STARVATION_LOG_PERSIST_FAILED", error=str(e))

        log_event("STARVATION_DAILY_SUMMARY",
                  evaluated=summary['total_evaluated'],
                  generated=summary['total_generated'],
                  executed=summary['total_executed'],
                  pass_rate=round(summary['pass_through_rate'], 3))

        return summary
