from fishing_bot.input_controller import InputController


def test_dry_run_reports_success_without_creating_hardware_controllers() -> None:
    controller = InputController(lambda: True, dry_run=True)
    assert controller.cast("0")
    assert controller.move_to(100, 200)
    assert controller.loot()
    assert controller._keyboard is None
    assert controller._mouse is None


def test_dry_run_still_obeys_safety_gate() -> None:
    controller = InputController(lambda: False, dry_run=True)
    assert not controller.cast("0")
    assert not controller.move_to(100, 200)
    assert not controller.loot()
