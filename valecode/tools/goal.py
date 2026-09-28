from valecode.tools.workflow import WorkflowTool


class GoalTool(WorkflowTool):
    name = "Goal"
    description = "仅在用户明确要求 Goal 目标闭环时调用，不从普通任务推断授权。run 加载工作区目标 YAML（objective/criteria/evidence_files/预算），独立只读验收与有界续跑。list/status 查看；resume 保留已执行阶段；retry 需用户确认重复副作用；cancel 中断。授权不足、缺证据、外部等待、无进展或预算耗尽均停止，不声称完成。"
