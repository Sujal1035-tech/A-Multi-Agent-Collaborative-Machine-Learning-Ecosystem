import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, Check, Loader2, Cpu, Brain, Layers, BarChart3, ShieldCheck, FileJson, FileCode, Workflow, Activity } from 'lucide-react';
import { streamAgentLogs } from '../services/api';

const AGENTS = [
    { id: 'analysis', label: 'Analysis', icon: Layers, desc: 'Scanning dataset structure' },
    { id: 'insight', label: 'Intelligence', icon: Brain, desc: 'Generating AI rationale' },
    { id: 'preprocessing', label: 'Cleaning', icon: Cpu, desc: 'Handling nulls/outliers' },
    { id: 'feature', label: 'Featurizer', icon: FileJson, desc: 'Encoding categorical data' },
    { id: 'model', label: 'Trainer', icon: BarChart3, desc: 'Optimizing ML algorithms' },
    { id: 'evaluation', label: 'Validator', icon: ShieldCheck, desc: 'Testing for overfitting' },
    { id: 'project', label: 'Architect', icon: FileCode, desc: 'Building source files' }
];

export default function ExecutionTerminal({ initialPayload, onPipelineComplete }) {
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [logs, setLogs] = useState([]);
    const [isFinished, setIsFinished] = useState(false);
    const stepDataRef = useRef({});
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    useEffect(() => {
        if (isFinished || currentStepIndex >= AGENTS.length) return;

        const currentStep = AGENTS[currentStepIndex];
        let payload = { ...initialPayload };
        const accumulated = stepDataRef.current;
        if (accumulated.analysis) payload.analysis_summary = accumulated.analysis;
        if (accumulated.preprocessing) payload.prep_strategy = accumulated.preprocessing;
        if (accumulated.feature) payload.feat_strategy = accumulated.feature;
        if (accumulated.model) payload.best_model_info = accumulated.model;

        const abortStream = streamAgentLogs(
            currentStep.id,
            payload,
            (logMsg) => {
                const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                setLogs(prev => [...prev, { type: 'log', text: logMsg, time: t, agent: currentStep.label }]);
            },
            (resultData) => {
                const actualData = resultData?.output ?? resultData;
                stepDataRef.current = { ...stepDataRef.current, [currentStep.id]: actualData };
                const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                setLogs(prev => [...prev, { type: 'success', text: `${currentStep.label} completed`, time: t }]);
            },
            (errorMsg) => {
                const t = new Date().toLocaleTimeString('en-US', { hour12: false });
                setLogs(prev => [...prev, { type: 'error', text: `${currentStep.label} error: ${errorMsg}`, time: t }]);
            },
            () => {
                if (currentStepIndex < AGENTS.length - 1) {
                    setCurrentStepIndex(prev => prev + 1);
                } else {
                    const accumulated = stepDataRef.current;
                    const now = new Date().toLocaleString();
                    let trace = "# 🧠 Agent Thinking & Decision Trace\n\n";
                    trace += `**Dataset:** \`${initialPayload.csv_path || 'uploaded file'}\`  \n`;
                    trace += `**Target Column:** \`${initialPayload.target_column || '?'}\`  \n`;
                    trace += `**Run Time:** ${now}\n\n`;
                    trace += "---\n\n";

                    if (accumulated.analysis) {
                        const analysis = accumulated.analysis.analysis_summary || accumulated.analysis || {};
                        trace += "## Step 1: Dataset Overview\n\n";
                        const shape = analysis.shape || [];
                        if (shape.length === 2) {
                            trace += `The analysis agent scanned the dataset and found **${shape[0]} rows** and **${shape[1]} columns**. `;
                        }
                        trace += `The target column to predict is **\`${analysis.target_column || '?'}\`**.\n\n`;
                    }

                    if (accumulated.preprocessing) {
                        const strategy = accumulated.preprocessing.preprocessing_strategy || accumulated.preprocessing || {};
                        trace += "---\n\n## Step 2: Preprocessing Agent — Why It Chose Each Technique\n\n";
                        const null_strat = strategy.null_strategy || {};
                        for (const [col, config] of Object.entries(null_strat)) {
                            const method = typeof config === 'object' ? (config.method || 'unknown') : config;
                            const reason = typeof config === 'object' ? (config.reason || 'No explicit reason was provided.') : 'Standard fallback applied.';
                            trace += `- **${col}:** The agent chose **${method}** imputation.\n`;
                            trace += `  *Why?* ${reason}\n\n`;
                        }
                    }

                    if (accumulated.feature) {
                        const feat = accumulated.feature.feature_strategy || accumulated.feature || {};
                        trace += "---\n\n## Step 3: Feature Engineering Agent — Encoding & Dropping Decisions\n\n";
                        const enc = feat.encoding_strategy || {};
                        const enc_reason = enc.reason || 'Categorical features were detected and encoded based on their cardinality.';
                        trace += `The feature agent analyzed categories and reasoned: *"${enc_reason}"*\n\n`;
                    }

                    if (accumulated.model) {
                        const models = accumulated.model || {};
                        trace += "---\n\n## Step 4: Model Training Agent — Strategy\n\n";
                        trace += `1. **Final Selection:** Selected **${models.best_model || '?'}** with a score of **${(models.best_score || 0).toFixed(4)}**.\n\n`;
                    }

                    accumulated.pipeline_trace = trace;
                    setTimeout(() => onPipelineComplete(accumulated), 1500);
                }
            }
        );
        return () => abortStream();
    }, [currentStepIndex, initialPayload, isFinished, onPipelineComplete]);

    const progress = ((currentStepIndex + (isFinished ? 1 : 0)) / AGENTS.length) * 100;

    return (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '3rem', padding: '2rem 0', minHeight: 'calc(100vh - 160px)' }}>

            {/* ── Top: Visual Progress Hub ────────── */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
                <div style={{ position: 'relative', width: '240px', height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '2.5rem' }}>
                    
                    {/* Pulsing Background Glow */}
                    <motion.div 
                        animate={{ scale: [1, 1.1, 1], opacity: [0.3, 0.5, 0.3] }}
                        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                        style={{ position: 'absolute', width: '180px', height: '180px', background: 'var(--accent-glow)', borderRadius: '50%', filter: 'blur(30px)' }}
                    />

                    {/* SVG Circular Progress */}
                    <svg width="240" height="240" style={{ transform: 'rotate(-90deg)', position: 'absolute' }}>
                        <circle cx="120" cy="120" r="110" stroke="rgba(255,255,255,0.03)" strokeWidth="4" fill="none" />
                        <motion.circle
                            cx="120" cy="120" r="110" stroke="var(--accent-light)" strokeWidth="6" fill="none"
                            strokeDasharray="691"
                            animate={{ strokeDashoffset: 691 - (691 * progress) / 100 }}
                            transition={{ duration: 1, ease: "circOut" }}
                            strokeLinecap="round"
                        />
                    </svg>

                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', zIndex: 1 }}>
                        <div style={{ fontSize: '3.5rem', fontWeight: 900, color: 'var(--text-primary)', letterSpacing: '-0.05em', lineHeight: 1 }}>
                            {Math.round(progress)}<span style={{ fontSize: '1.2rem', color: 'var(--accent-light)', verticalAlign: 'top', marginLeft: '2px' }}>%</span>
                        </div>
                        <div className="premium-tag" style={{ marginTop: '0.75rem', transform: 'scale(0.85)' }}>
                             Active Engine
                        </div>
                    </div>
                </div>

                <div className="stagger-1">
                    <h2 style={{ fontSize: '1.75rem', fontWeight: 900, color: 'var(--text-primary)', marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>
                       Orchestrating Intelligence
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', maxWidth: '440px', fontWeight: 500 }}>
                        Seven specialized AI agents are currently synchronizing to transform your dataset into a production-ready model.
                    </p>
                </div>
            </div>

            {/* ── Mid: Agent Timeline ─────────────── */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '1rem', width: '100%', maxWidth: '1100px', margin: '0 auto' }}>
                {AGENTS.map((agent, i) => {
                    const isActive = i === currentStepIndex;
                    const isDone = i < currentStepIndex;
                    const Icon = agent.icon;

                    return (
                        <div key={agent.id} className={`stagger-${(i%4)+1}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.25rem', opacity: isActive || isDone ? 1 : 0.3, transition: 'all 0.4s' }}>
                            <div style={{
                                width: '64px', height: '64px', borderRadius: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: isActive ? 'var(--accent-glow)' : 'rgba(255,255,255,0.02)',
                                border: isActive ? '1px solid var(--accent-light)' : '1px solid var(--glass-border)',
                                position: 'relative',
                                boxShadow: isActive ? '0 0 30px rgba(16,185,129,0.2)' : 'none'
                             }}>
                                <Icon size={26} color={isActive || isDone ? 'var(--accent-light)' : 'var(--text-muted)'} />
                                {isActive && (
                                    <motion.div
                                        layoutId="pulse"
                                        style={{ position: 'absolute', inset: -1, borderRadius: '20px', border: '1px solid var(--accent-light)', opacity: 0.5 }}
                                        animate={{ scale: [1, 1.25, 1], opacity: [0.5, 0, 0.5] }}
                                        transition={{ duration: 1.5, repeat: Infinity }}
                                    />
                                )}
                                {isDone && (
                                    <div style={{ position: 'absolute', bottom: -5, right: -5, background: 'var(--accent-light)', borderRadius: '50%', padding: '0.2rem', boxShadow: '0 2px 8px rgba(0,0,0,0.4)' }}>
                                        <Check size={10} color="#000" strokeWidth={4} />
                                    </div>
                                )}
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: '0.8rem', fontWeight: 800, color: isActive ? 'var(--accent-light)' : 'var(--text-primary)', letterSpacing: '-0.01em' }}>{agent.label}</div>
                                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 500, marginTop: '2px', display: isActive ? 'block' : 'none' }}>{agent.desc}</div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* ── Bottom: Stream Feed ──────────────── */}
            <div className="card stagger-4" style={{ maxWidth: '800px', width: '100%', margin: '0 auto', height: '280px', display: 'flex', flexDirection: 'column', padding: '0', border: '1px solid var(--glass-border)', overflow: 'hidden', boxShadow: '0 20px 50px rgba(0,0,0,0.5)' }}>
                 <div style={{ padding: '0.85rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.03)', background: 'rgba(255,255,255,0.01)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        <Workflow size={14} color="var(--accent-light)" /> System Diagnostics
                    </div>
                </div>
                <div style={{ flex: 1, padding: '1.25rem', overflowY: 'auto', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.8rem', background: 'rgba(0,0,0,0.3)', color: 'rgba(248,250,252,0.8)' }}>
                    <AnimatePresence mode="popLayout">
                        {logs.slice(-50).map((log, i) => (
                            <motion.div
                                key={`${i}-${log.time}`}
                                initial={{ opacity: 0, y: 5 }}
                                animate={{ opacity: 1, y: 0 }}
                                style={{
                                    display: 'flex', gap: '1rem', marginBottom: '0.6rem',
                                    padding: '0.5rem 0.75rem', borderRadius: '6px',
                                    borderLeft: log.type === 'success' ? '2px solid var(--accent-light)' : log.type === 'error' ? '2px solid var(--danger)' : '2px solid transparent',
                                    background: log.type === 'success' ? 'rgba(16,185,129,0.03)' : log.type === 'error' ? 'rgba(239,68,68,0.03)' : 'transparent'
                                }}
                            >
                                <span style={{ opacity: 0.2, flexShrink: 0, fontWeight: 500 }}>{log.time}</span>
                                {log.agent && <span style={{ color: 'var(--accent-light)', fontWeight: 800, minWidth: '94px', fontSize: '0.7rem' }}>[{log.agent.toUpperCase()}]</span>}
                                <span style={{ color: log.type === 'success' ? 'var(--accent-light)' : log.type === 'error' ? 'var(--danger)' : 'inherit', fontWeight: 400, lineHeight: 1.5 }}>
                                    {log.text}
                                </span>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    <div ref={bottomRef} />
                </div>
            </div>
        </div>
    );
}
