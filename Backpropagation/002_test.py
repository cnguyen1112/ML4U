import importlib
import math


def run_tests():
    mod = importlib.import_module("002")

    # 1. Test ham du doan: cost(X, w, b)
    preds = mod.cost([1.0, 2.0, 3.0, 4.0], 2.0, 1.0)
    assert preds == [3.0, 5.0, 7.0, 9.0], f"cost() sai: {preds}"

    # 2. Test ham loss(Y_pred, Y)
    assert math.isclose(mod.loss([3.0, 5.0], [3.0, 5.0]), 0.0)
    assert math.isclose(mod.loss([0.0, 0.0, 0.0, 0.0], [3.0, 5.0, 7.0, 9.0]), 41.0)

    # 3. Test grad_w va grad_b tai w=0.0, b=0.0
    X = [1.0, 2.0, 3.0, 4.0]
    Y = [3.0, 5.0, 7.0, 9.0]
    Y_pred = [0.0, 0.0, 0.0, 0.0]
    dw = mod.grad_w(X, Y_pred, Y)
    db = mod.grad_b(Y_pred, Y)
    assert math.isclose(dw, -35.0), f"grad_w() sai: {dw} (ky vong -35.0)"
    assert math.isclose(db, -12.0), f"grad_b() sai: {db} (ky vong -12.0)"

    # 4. Test training_func(X, Y, w, b, lr)
    result = mod.training_func(X, Y, 0.0, 0.0, 0.01)
    assert result is not None, (
        "training_func() dang tra ve None! Hay them `return w, b` o cuoi ham."
    )
    w_final, b_final = result
    final_loss = mod.loss(mod.cost(X, w_final, b_final), Y)

    print(f"\nKet qua sau huan luyen: w = {w_final:.6f}, b = {b_final:.6f}, loss = {final_loss:.6f}")
    assert abs(w_final - 2.0) < 0.05, f"w chua hoi tu ve 2.0: {w_final}"
    assert abs(b_final - 1.0) < 0.1, f"b chua hoi tu ve 1.0: {b_final}"
    assert final_loss < 1e-2, f"loss chua du nho: {final_loss}"

    print("TAT CA TEST BAI 2 DEU PASS!")


if __name__ == "__main__":
    run_tests()
