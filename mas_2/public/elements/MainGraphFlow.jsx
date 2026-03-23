import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Bot,
  Code2,
  Flag,
  GitBranch,
  Search,
  ShieldCheck,
  Wrench,
} from "lucide-react";

const ICONS = {
  supervisor: GitBranch,
  rag_researcher: Search,
  code_dev: Code2,
  tool_caller: Wrench,
  critic: ShieldCheck,
  finalize: Flag,
};

const SHORT_LABEL = {
  supervisor: "Supervisor",
  rag_researcher: "RAG",
  code_dev: "Code Dev",
  tool_caller: "Tools",
  critic: "Critic",
  finalize: "Finalize",
};

function AgentCard({
  id,
  label,
  Icon,
  phase,
  onSelect,
  selected,
  activeId,
  visitedSequence,
}) {
  const isActive = activeId === id;
  const isDone = visitedSequence && visitedSequence.includes(id);
  const isIdle = !isActive && !isDone;

  let ring = "border-border/80 bg-card/40";
  if (isIdle) {
    ring = "border-dashed border-muted-foreground/25 bg-muted/20 opacity-90";
  } else if (isDone && !isActive) {
    ring = "border-emerald-500/35 bg-card shadow-sm";
  }
  if (isActive) {
    ring =
      "border-primary/60 bg-primary/5 shadow-md ring-2 ring-primary/70 ring-offset-2 ring-offset-background transition-shadow duration-300";
  }

  return (
    <button
      type="button"
      onClick={() => onSelect(id)}
      className={`rounded-xl border text-left transition-all duration-200 ${ring} ${
        selected ? "outline outline-2 outline-offset-2 outline-primary/50" : ""
      } hover:border-primary/40 hover:bg-accent/30`}
    >
      <div className="flex items-center gap-2 px-3 py-2.5">
        <span
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
            isActive
              ? "bg-primary/15 text-primary"
              : isDone
                ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                : "bg-muted text-muted-foreground"
          }`}
        >
          <Icon className="h-4 w-4" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold leading-tight">{label}</div>
          {phase ? (
            <div className="mt-0.5 truncate text-[10px] text-muted-foreground">
              {phase}
            </div>
          ) : null}
        </div>
        {isActive ? (
          <span className="shrink-0 rounded-full bg-primary px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide text-primary-foreground">
            运行
          </span>
        ) : isDone ? (
          <span className="shrink-0 text-[9px] font-medium text-emerald-600 dark:text-emerald-400">
            已执行
          </span>
        ) : (
          <span className="shrink-0 text-[9px] text-muted-foreground">待命中</span>
        )}
      </div>
    </button>
  );
}

export default function MainGraphFlow() {
  const agentOrder = props.agentOrder || [
    "supervisor",
    "rag_researcher",
    "code_dev",
    "tool_caller",
    "critic",
    "finalize",
  ];
  const snapshots = props.snapshots || {};
  const activeId = props.activeId ?? null;
  const visitedSequence = props.visitedSequence || [];
  const statusLine = props.statusLine || "";
  const pathLine = props.pathLine || "";
  const note = props.note || "";

  const [selectedId, setSelectedId] = useState(null);

  useEffect(() => {
    if (activeId) {
      setSelectedId(activeId);
    }
  }, [activeId]);

  const detailSnap =
    selectedId && snapshots[selectedId] ? snapshots[selectedId] : null;

  return (
    <div className="flex w-full max-w-md flex-col gap-3">
      <div className="space-y-1 rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent px-3 py-2">
        <div className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <Bot className="h-4 w-4 text-primary" aria-hidden />
          主图执行
        </div>
        {statusLine ? (
          <p className="whitespace-pre-wrap text-xs text-foreground/90">
            {statusLine}
          </p>
        ) : null}
        {pathLine ? (
          <p className="text-[11px] leading-snug text-muted-foreground">
            {pathLine}
          </p>
        ) : null}
        {note ? (
          <p className="text-[11px] text-amber-700 dark:text-amber-500">
            {note}
          </p>
        ) : null}
      </div>

      <div className="grid grid-cols-1 gap-2">
        {agentOrder[0] ? (
          <AgentCard
            id={agentOrder[0]}
            label={SHORT_LABEL[agentOrder[0]] || agentOrder[0]}
            Icon={ICONS[agentOrder[0]] || Bot}
            phase={
              snapshots[agentOrder[0]]?.innerLine ||
              (activeId === agentOrder[0] ? "…" : null)
            }
            onSelect={setSelectedId}
            selected={selectedId === agentOrder[0]}
            activeId={activeId}
            visitedSequence={visitedSequence}
          />
        ) : null}

        <div className="grid grid-cols-3 gap-2">
          {[1, 2, 3].map((i) => {
            const id = agentOrder[i];
            if (!id) return null;
            return (
              <AgentCard
                key={id}
                id={id}
                label={SHORT_LABEL[id] || id}
                Icon={ICONS[id] || Bot}
                phase={
                  snapshots[id]?.innerLine ||
                  (activeId === id ? "…" : null)
                }
                onSelect={setSelectedId}
                selected={selectedId === id}
                activeId={activeId}
                visitedSequence={visitedSequence}
              />
            );
          })}
        </div>

        {agentOrder[4] ? (
          <AgentCard
            id={agentOrder[4]}
            label={SHORT_LABEL[agentOrder[4]] || agentOrder[4]}
            Icon={ICONS[agentOrder[4]] || Bot}
            phase={
              snapshots[agentOrder[4]]?.innerLine ||
              (activeId === agentOrder[4] ? "…" : null)
            }
            onSelect={setSelectedId}
            selected={selectedId === agentOrder[4]}
            activeId={activeId}
            visitedSequence={visitedSequence}
          />
        ) : null}

        {agentOrder[5] ? (
          <AgentCard
            id={agentOrder[5]}
            label={SHORT_LABEL[agentOrder[5]] || agentOrder[5]}
            Icon={ICONS[agentOrder[5]] || Bot}
            phase={
              snapshots[agentOrder[5]]?.innerLine ||
              (activeId === agentOrder[5] ? "…" : null)
            }
            onSelect={setSelectedId}
            selected={selectedId === agentOrder[5]}
            activeId={activeId}
            visitedSequence={visitedSequence}
          />
        ) : null}
      </div>

      <Card className="border-border/70 bg-card/80">
        <CardHeader className="space-y-0 border-b border-border/50 px-3 py-2">
          <div className="text-xs font-semibold">
            {selectedId
              ? `Agent 详情 · ${SHORT_LABEL[selectedId] || selectedId}`
              : "Agent 详情"}
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-44 w-full rounded-b-lg">
            <div className="space-y-2 p-3 text-xs leading-relaxed">
              {!selectedId ? (
                <p className="text-muted-foreground">点击上方卡片查看摘要。</p>
              ) : detailSnap ? (
                <>
                  {detailSnap.innerLine ? (
                    <div className="rounded-md border border-primary/20 bg-primary/5 px-2 py-1.5 text-[11px] font-medium text-primary">
                      子阶段：{detailSnap.innerLine}
                    </div>
                  ) : null}
                  <div className="font-medium text-foreground/90">
                    {detailSnap.title}
                  </div>
                  <ul className="list-disc space-y-1.5 pl-4 text-muted-foreground">
                    {(detailSnap.lines || []).map((line, idx) => (
                      <li key={idx} className="break-words">
                        {line}
                      </li>
                    ))}
                  </ul>
                </>
              ) : activeId === selectedId ? (
                <p className="text-primary">
                  该节点正在运行，完成后将显示状态摘要…
                </p>
              ) : (
                <p className="text-muted-foreground">
                  该 Agent 本轮尚未产生状态摘要。
                </p>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
