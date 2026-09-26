from valecode.commands.registry import Command, CommandType
from valecode.tools.cron import CronList, EmptyParams


def create_cron_command(runtime):
    async def handler(ctx):
        parts = ctx.args.strip().split()
        try:
            session_id = getattr(ctx.session, "session_id", "")
            if not session_id or session_id != runtime.agent_tool._parent_agent.session_id:
                raise ValueError("没有活动会话")
            if not parts or parts == ["list"]:
                result = await CronList(runtime).execute(EmptyParams())
                ctx.ui.add_system_message(result.output)
            elif len(parts) == 2 and parts[0] in {"pause", "resume", "delete"}:
                plan = runtime.store.change(parts[1], session_id, parts[0])
                ctx.ui.add_system_message(f'{plan["id"]}: {plan["status"]}')
            else:
                ctx.ui.add_system_message("用法: /cron [list|pause|resume|delete] [schedule-id]")
        except (ValueError, KeyError) as exc:
            ctx.ui.add_system_message(str(exc))
    return Command(name="cron", description="管理当前会话的定时计划", type=CommandType.LOCAL,
        handler=handler, usage="/cron [list|pause|resume|delete] [schedule-id]")
