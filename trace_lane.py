"""Trace select_square for cycle with unknown cards — match the live path."""
from adapters.clash_adapter import TacticalReflexPolicy
from clash_jev.moves import legal_squares
from clash_jev.state import BattleState, HandCard, Towers
from clash_jev.cards import info


class FakeLane:
    enemy_on_my_side = 0
    enemy_at_bridge = 0
    enemy_on_their_side = 0
    mine_on_my_side = 0
    mine_at_bridge = 0
    mine_on_their_side = 0


def make_state(elixir: int, hand) -> BattleState:
    return BattleState(
        elapsed_s=30.0,
        elixir=elixir,
        hand=tuple(hand),
        left=FakeLane(),
        right=FakeLane(),
        towers=Towers(),
        units=(),
        snapshot_interval_s=1.0,
        seconds_at_full_elixir=0.0,
    )


def main() -> None:
    card = HandCard(slot=0, name="unknown", ready=True)
    state = make_state(elixir=7, hand=[card])
    print("kind", info("unknown").kind)
    sqs = legal_squares(card, state)
    print("legal:", [s.name for s in sqs])

    policy = TacticalReflexPolicy()
    names = []
    for i in range(7):
        sq = policy.select_square("cycle", card, state)
        names.append(sq.name if sq else None)
        print(f"  call {i}: counter={policy._push_counter} -> {names[-1]}")
    print("sequence:", names)

    # Also try knight for comparison
    card2 = HandCard(slot=0, name="knight", ready=True)
    state2 = make_state(elixir=8, hand=[card2])
    policy2 = TacticalReflexPolicy()
    names2 = []
    for i in range(7):
        sq = policy2.select_square("cycle", card2, state2)
        names2.append(sq.name if sq else None)
    print("knight sequence:", names2)


if __name__ == "__main__":
    main()
