import torch
from net import ActorCritic   # 改成你的文件名


def test_model_io():
    # 1. 超参数
    batch_size = 4
    agent_num = 10
    agent_dim = 6
    task_num = 16   # depot + 15 tasks
    task_dim = 5
    hidden_dim = 128

    # 2. 构造假输入
    agents_info = torch.randn(batch_size, agent_num, agent_dim)   # (B, 10, 6)
    task_info = torch.randn(batch_size, task_num, task_dim)       # (B, 17, 5)

    # mask=True 表示该动作不可选
    mask_info = torch.zeros(batch_size, task_num, dtype=torch.bool)

    # 随机屏蔽几个动作做测试
    mask_info[0, 3] = True
    mask_info[1, 5] = True
    mask_info[2, 0] = True   # depot 也可以测试一下
    mask_info[3, 10] = True

    # 3. 初始化网络
    model = ActorCritic(agent_dim=agent_dim, task_dim=task_dim, hidden_dim=hidden_dim)

    # 4. 前向传播
    logits, value = model(agents_info, task_info, mask_info)

    # 5. 打印输出 shape
    print("agents_info shape:", agents_info.shape)
    print("task_info shape:  ", task_info.shape)
    print("mask_info shape:  ", mask_info.shape)
    print("logits shape:     ", logits.shape)
    print("value shape:      ", value.shape)

    # 6. 检查 shape 是否正确
    assert logits.shape == (batch_size, task_num), f"logits shape error: {logits.shape}"
    assert value.shape == (batch_size, 1), f"value shape error: {value.shape}"

    # 7. 检查 mask 后的 logits 是否足够小
    masked_logits = logits[mask_info]
    if masked_logits.numel() > 0:
        print("masked logits:", masked_logits)
        assert torch.all(masked_logits < -1e8), "mask 没有正确生效"

    # 8. 测试 act 接口
    action, log_prob, entropy, value2 = model.act(
        agents_info, task_info, mask_info, deterministic=False
    )

    print("action shape:     ", action.shape)
    print("log_prob shape:   ", log_prob.shape)
    print("entropy shape:    ", entropy.shape)
    print("value2 shape:     ", value2.shape)

    assert action.shape == (batch_size,), f"action shape error: {action.shape}"
    assert log_prob.shape == (batch_size,), f"log_prob shape error: {log_prob.shape}"
    assert entropy.shape == (batch_size,), f"entropy shape error: {entropy.shape}"
    assert value2.shape == (batch_size,), f"value2 shape error: {value2.shape}"

    # 9. 检查采样动作是否落在合法范围内
    assert torch.all(action >= 0) and torch.all(action < task_num), "action 超出范围"

    print("\n模型输入输出测试通过。")


if __name__ == "__main__":
    test_model_io()