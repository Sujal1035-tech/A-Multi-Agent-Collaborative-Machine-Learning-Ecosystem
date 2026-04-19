import { useState, useEffect, useMemo } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import {
    Download, CheckCircle, Activity, Zap, Layers,
    ChevronDown, Cpu, Database, Trophy, Eye, EyeOff,
    Sparkles, Crown, Medal, Award,
    ExternalLink, Code2, FileDown, Copy, Check
} from 'lucide-react';

const API_BASE = 'http://localhost:8081';

/*  ── Micro Score Gauge ─────────────────────────────── */
function ScoreGauge({ value, size = 72, label }) {
    const pct = Math.min(Math.max(value * 100, 0), 100);
    const r = (size - 8) / 2;
    const circ = 2 * Math.PI * r;
    const offset = circ - (circ * pct) / 100;
    const color = pct >= 90 ? '#10b981' : pct >= 75 ? '#34d399' : pct >= 50 ? '#f59e0b' : '#ef4444';

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
            <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
                <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="4" />
                <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth="4"
                    strokeDasharray={circ} strokeDashoffset={offset}
                    strokeLinecap="round" style={{ transition: 'stroke-dashoffset 1s ease' }}
                />
            </svg>
            <div style={{ position: 'relative', marginTop: -size - 4, height: size, width: size, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ fontSize: size * 0.22, fontWeight: 900, color: '#f8fafc' }}>{pct.toFixed(1)}</span>
            </div>
            {label && <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'rgba(148,163,184,0.8)', textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: '0.15rem' }}>{label}</div>}
        </div>
    );
}

/* ── Copy Button ─────────────────────────────────── */
function CopyBtn({ text }) {
    const [copied, setCopied] = useState(false);
    return (
        <button className="btn btn-ghost btn-sm" onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
            style={{ gap: '0.3rem' }}>
            {copied ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
        </button>
    );
}

/* ────────────────────────────────────────────────── */
export default function ResultsDashboard({ results, runId }) {
    const [activeTab, setActiveTab] = useState('overview');
    const [summaryExpanded, setSummaryExpanded] = useState(false);
    const [selectedModel, setSelectedModel] = useState(null);

    // ── Data extraction ──
    const analysisOutput = results?.analysis || {};
    const analysisSummary = analysisOutput.analysis_summary || {};
    const insightOutput = results?.insight || results?.insights_2 || results?.insights_1 || {};
    const insightText = insightOutput.insights || insightOutput.insight_report || insightOutput.summary || '';
    const modelOutput = results?.model || {};
    const evalOutput = results?.evaluation || {};
    const evalData = evalOutput.evaluation || {};
    const evalReport = typeof evalData === 'string' ? evalData : (evalData.analysis_summary || '');
    const projectOutput = results?.project || {};
    const generatedCode = projectOutput.analysis_code || projectOutput.code || '';
    const readmeContent = projectOutput.readme || '';

    const shape = analysisSummary.shape || [];
    const bestModelName = modelOutput.best_model || 'N/A';
    const problemType = modelOutput.problem_type || 'classification';
    const isClassification = problemType === 'classification';
    const allModels = modelOutput.models || {};
    const bestModelData = allModels[bestModelName] || {};
    const bestScore = isClassification ? (bestModelData.accuracy || bestModelData.score || 0) : (bestModelData.r2 || bestModelData.score || 0);

    const rawImportances = modelOutput.shap_values || modelOutput.feature_importance || [];
    const featureImportances = Array.isArray(rawImportances)
        ? rawImportances.map(item => ({ name: item.feature ?? item[0] ?? '?', value: Math.round((item.importance ?? item[1] ?? 0) * 1000) / 1000 }))
        : Object.entries(rawImportances).map(([name, value]) => ({ name, value: Math.round(value * 1000) / 1000 }));

    const plotFiles = ['correlation_heatmap.png', 'target_distribution.png', 'feature_distributions.png', 'box_plots.png', 'shap_summary.png'];
    const [availablePlots, setAvailablePlots] = useState([]);

    useEffect(() => {
        if (!runId) return;
        (async () => {
            const avail = [];
            for (const pf of plotFiles) {
                try {
                    const url = `${API_BASE}/static/outputs/${runId}/plots/${pf}`;
                    if ((await fetch(url, { method: 'HEAD' })).ok) avail.push({ name: pf.replace('.png', '').replace(/_/g, ' '), url });
                } catch { }
            }
            setAvailablePlots(avail);
        })();
    }, [runId]);

    const modelCards = useMemo(() =>
        Object.entries(allModels)
            .filter(([_, m]) => (m.score || 0) > 0)
            .map(([name, m]) => ({ id: name, name: name.replace(/_/g, ' '), ...m }))
            .sort((a, b) => (b.score || 0) - (a.score || 0)),
        [allModels]
    );

    const fmt = (v, mode = 'num') => {
        if (v == null || isNaN(v)) return '—';
        if (mode === 'pct') return `${(v * 100).toFixed(2)}%`;
        return v.toFixed(4);
    };

    const radarData = isClassification
        ? [
            { m: 'Accuracy', v: (bestModelData.accuracy || 0) * 100 },
            { m: 'Precision', v: (bestModelData.precision || 0) * 100 },
            { m: 'Recall', v: (bestModelData.recall || 0) * 100 },
            { m: 'F1', v: (bestModelData.f1_score || 0) * 100 },
            { m: 'AUC', v: (bestModelData.auc_roc || 0) * 100 },
            { m: 'CV', v: (bestModelData.cv_score || 0) * 100 },
        ].filter(d => d.v > 0)
        : [];

    const tooltipStyle = {
        backgroundColor: 'rgba(2, 6, 23, 0.95)',
        border: '1px solid rgba(52,211,153,0.15)',
        borderRadius: '10px',
        padding: '10px 14px',
        boxShadow: '0 12px 40px rgba(0,0,0,0.6)',
        fontSize: '0.78rem',
        color: '#e2e8f0',
    };

    const handleDownloadZip = async () => {
        try {
            const JSZip = (await import('jszip')).default;
            const zip = new JSZip();
            if (generatedCode) zip.file('analysis.py', generatedCode);
            if (readmeContent) zip.file('README.md', readmeContent);
            if (insightText) zip.file('insights.txt', insightText);
            if (evalReport) zip.file('reports/evaluation.txt', evalReport);
            const traceText = results?.pipeline_trace || '';
            if (traceText) zip.file('pipeline_trace.md', traceText);
            for (const plot of availablePlots) {
                try { const res = await fetch(plot.url); const blob = await res.blob(); zip.file(`plots/${plot.name.replace(/ /g, '_')}.png`, blob); } catch { }
            }
            const blob = await zip.generateAsync({ type: 'blob' });
            const url = URL.createObjectURL(blob);
            Object.assign(document.createElement('a'), { href: url, download: `${runId || 'autoeda'}.zip` }).click();
            URL.revokeObjectURL(url);
        } catch { alert('Download failed'); }
    };

    const TABS = [
        { id: 'overview', label: 'Overview' },
        { id: 'models', label: `Models (${modelCards.length})` },
        { id: 'visuals', label: 'Visualisations' },
        { id: 'export', label: 'Export' },
    ];

    const rankIcon = (i) => {
        if (i === 0) return <Crown size={14} style={{ color: '#fbbf24' }} />;
        if (i === 1) return <Medal size={14} style={{ color: '#94a3b8' }} />;
        if (i === 2) return <Award size={14} style={{ color: '#b45309' }} />;
        return <span style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-muted)', width: 14, textAlign: 'center', display: 'inline-block' }}>#{i + 1}</span>;
    };

    return (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 0, paddingBottom: '4rem' }}>

            {/* ══════════  METRIC RIBBON  ══════════ */}
            <div className="stagger-1" style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(5, 1fr)',
                background: 'rgba(10, 10, 14, 0.8)',
                border: '1px solid var(--glass-border)',
                borderRadius: '16px',
                marginBottom: '2rem',
                overflow: 'hidden'
            }}>
                {[
                    { label: 'Champion', value: bestModelName.replace(/_/g, ' '), accent: true, icon: <Crown size={14} style={{ color: '#fbbf24' }} /> },
                    { label: isClassification ? 'Accuracy' : 'R² Score', value: fmt(bestScore, 'pct'), accent: true },
                    { label: 'Rows', value: shape[0]?.toLocaleString() || '—' },
                    { label: 'Features', value: shape[1] || '—' },
                    { label: 'Target', value: results?.analysis?.analysis_summary?.target_column || '?', accent: true },
                ].map((cell, i) => (
                    <div key={i} style={{
                        padding: '1.25rem 1.5rem',
                        borderRight: i < 4 ? '1px solid var(--glass-border)' : 'none',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '0.35rem'
                    }}>
                        <div style={{ fontSize: '0.62rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.12em', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            {cell.icon}{cell.label}
                        </div>
                        <div style={{
                            fontSize: cell.accent ? '1.15rem' : '1rem',
                            fontWeight: 900,
                            color: cell.accent ? 'var(--accent-light)' : 'var(--text-primary)',
                            letterSpacing: '-0.02em',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis'
                        }}>
                            {cell.value}
                        </div>
                    </div>
                ))}
            </div>

            {/* ══════════  NAVIGATION TABS  ══════════ */}
            <div className="stagger-2" style={{
                display: 'flex',
                gap: '0.25rem',
                marginBottom: '2rem',
                borderBottom: '1px solid var(--glass-border)',
                paddingBottom: '0'
            }}>
                {TABS.map(tab => (
                    <button key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        style={{
                            padding: '0.75rem 1.5rem',
                            fontSize: '0.82rem',
                            fontWeight: activeTab === tab.id ? 800 : 600,
                            color: activeTab === tab.id ? 'var(--accent-light)' : 'var(--text-muted)',
                            background: 'none',
                            border: 'none',
                            borderBottom: activeTab === tab.id ? '2px solid var(--accent-light)' : '2px solid transparent',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            fontFamily: 'inherit',
                            marginBottom: '-1px'
                        }}>
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* ══════════  TAB: OVERVIEW  ══════════ */}
            {activeTab === 'overview' && (
                <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

                    {/* Key metrics row */}
                    {isClassification ? (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '1rem' }}>
                            {[
                                { k: 'Accuracy', v: bestModelData.accuracy },
                                { k: 'Precision', v: bestModelData.precision },
                                { k: 'Recall', v: bestModelData.recall },
                                { k: 'F1 Score', v: bestModelData.f1_score },
                                { k: 'AUC-ROC', v: bestModelData.auc_roc },
                                { k: 'CV Mean', v: bestModelData.cv_score },
                            ].map(m => (
                                <div key={m.k} style={{
                                    background: 'rgba(255,255,255,0.015)',
                                    border: '1px solid var(--glass-border)',
                                    borderRadius: '12px',
                                    padding: '1.25rem 1rem',
                                    textAlign: 'center'
                                }}>
                                    <div style={{ fontSize: '0.6rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>{m.k}</div>
                                    <div style={{ fontSize: '1.3rem', fontWeight: 900, color: m.v != null && m.v > 0.85 ? 'var(--accent-light)' : 'var(--text-primary)' }}>
                                        {fmt(m.v, 'pct')}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '1rem' }}>
                            {[
                                { k: 'R² Score', v: bestModelData.r2, mode: 'num' },
                                { k: 'Adj. R²', v: bestModelData.adjusted_r2, mode: 'num' },
                                { k: 'RMSE', v: bestModelData.rmse, mode: 'num' },
                                { k: 'MAE', v: bestModelData.mae, mode: 'num' },
                                { k: 'CV Mean', v: bestModelData.cv_score, mode: 'num' },
                            ].map(m => (
                                <div key={m.k} style={{
                                    background: 'rgba(255,255,255,0.015)',
                                    border: '1px solid var(--glass-border)',
                                    borderRadius: '12px',
                                    padding: '1.25rem 1rem',
                                    textAlign: 'center'
                                }}>
                                    <div style={{ fontSize: '0.6rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>{m.k}</div>
                                    <div style={{ fontSize: '1.3rem', fontWeight: 900, color: 'var(--text-primary)' }}>{fmt(m.v, m.mode)}</div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Two-column: Radar + Summary */}
                    <div style={{ display: 'grid', gridTemplateColumns: isClassification && radarData.length > 0 ? '1fr 1.6fr' : '1fr', gap: '1.5rem' }}>

                        {isClassification && radarData.length > 0 && (
                            <div style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid var(--glass-border)', borderRadius: '16px', padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
                                <div style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1rem' }}>
                                    Performance Profile
                                </div>
                                <div style={{ flex: 1, minHeight: '300px' }}>
                                    <ResponsiveContainer>
                                        <RadarChart data={radarData}>
                                            <PolarGrid stroke="rgba(255,255,255,0.04)" />
                                            <PolarAngleAxis dataKey="m" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 700 }} />
                                            <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                                            <defs>
                                                <radialGradient id="rg" cx="50%" cy="50%" r="50%">
                                                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.5} />
                                                    <stop offset="100%" stopColor="#10b981" stopOpacity={0.05} />
                                                </radialGradient>
                                            </defs>
                                            <Radar dataKey="v" stroke="#34d399" fill="url(#rg)" fillOpacity={0.8} strokeWidth={2} dot={{ r: 3, fill: '#34d399' }} />
                                            <Tooltip contentStyle={tooltipStyle} />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* Executive Summary */}
                        <div style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid var(--glass-border)', borderRadius: '16px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                            <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.03)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                    <Sparkles size={16} style={{ color: 'var(--accent-light)' }} />
                                    <span style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>AI Insights</span>
                                </div>
                                <button onClick={() => setSummaryExpanded(!summaryExpanded)}
                                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--accent-light)', fontSize: '0.72rem', fontWeight: 700, fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                    {summaryExpanded ? <><EyeOff size={13} /> Less</> : <><Eye size={13} /> Full Report</>}
                                </button>
                            </div>
                            <div style={{
                                padding: '1.5rem',
                                maxHeight: summaryExpanded ? '3000px' : '320px',
                                overflow: 'hidden',
                                position: 'relative',
                                transition: 'max-height 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
                            }}>
                                <div style={{ color: 'var(--text-secondary)', lineHeight: 1.85, fontSize: '0.92rem', whiteSpace: 'pre-wrap' }}>
                                    {insightText || 'Generating strategic insights...'}
                                    {evalReport && (
                                        <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.03)' }}>
                                            <div style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--accent-light)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                                <CheckCircle size={12} /> Quality Assessment
                                            </div>
                                            {evalReport}
                                        </div>
                                    )}
                                </div>
                                {!summaryExpanded && (
                                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '100px', background: 'linear-gradient(transparent, rgba(10,10,14,1))', pointerEvents: 'none' }} />
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Feature Importance */}
                    {featureImportances.length > 0 && (
                        <div style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid var(--glass-border)', borderRadius: '16px', padding: '1.5rem' }}>
                            <div style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1.5rem' }}>
                                Feature Drivers (SHAP)
                            </div>
                            <div style={{ height: Math.max(250, featureImportances.slice(0, 8).length * 38) }}>
                                <ResponsiveContainer>
                                    <BarChart data={featureImportances.slice(0, 8)} layout="vertical" margin={{ left: 30, right: 30, top: 5, bottom: 5 }}>
                                        <defs>
                                            <linearGradient id="bg" x1="0" y1="0" x2="1" y2="0">
                                                <stop offset="0%" stopColor="#10b981" stopOpacity={0.6} />
                                                <stop offset="100%" stopColor="#34d399" stopOpacity={0.9} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.02)" horizontal={false} />
                                        <XAxis type="number" hide />
                                        <YAxis dataKey="name" type="category" stroke="#475569" fontSize={11} fontWeight={700} axisLine={false} tickLine={false} width={110} />
                                        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                                        <Bar dataKey="value" fill="url(#bg)" radius={[0, 6, 6, 0]} barSize={18} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ══════════  TAB: MODELS  ══════════ */}
            {activeTab === 'models' && (
                <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    {modelCards.map((m, idx) => {
                        const isBest = idx === 0;
                        const isOpen = selectedModel === m.id;
                        return (
                            <div key={m.id}
                                onClick={() => setSelectedModel(isOpen ? null : m.id)}
                                style={{
                                    background: isBest ? 'rgba(16,185,129,0.04)' : 'rgba(255,255,255,0.015)',
                                    border: isBest ? '1px solid rgba(16,185,129,0.2)' : '1px solid var(--glass-border)',
                                    borderRadius: '16px',
                                    padding: '1.25rem 1.75rem',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s',
                                }}>
                                {/* Main row */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                    <div style={{ width: '34px', display: 'flex', justifyContent: 'center' }}>
                                        {rankIcon(idx)}
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                            <span style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--text-primary)', textTransform: 'capitalize' }}>{m.name}</span>
                                            {isBest && <span style={{ fontSize: '0.55rem', fontWeight: 800, color: '#fbbf24', background: 'rgba(251,191,36,0.1)', padding: '0.15rem 0.5rem', borderRadius: '4px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Champion</span>}
                                        </div>
                                    </div>

                                    {/* Score pills */}
                                    <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
                                        {isClassification ? (
                                            <>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Accuracy</div>
                                                    <div style={{ fontSize: '1.1rem', fontWeight: 900, color: isBest ? 'var(--accent-light)' : 'var(--text-primary)' }}>{fmt(m.accuracy || m.score, 'pct')}</div>
                                                </div>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>F1</div>
                                                    <div style={{ fontSize: '1.1rem', fontWeight: 900, color: 'var(--text-primary)' }}>{fmt(m.f1_score, 'pct')}</div>
                                                </div>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>CV</div>
                                                    <div style={{ fontSize: '1.1rem', fontWeight: 900, color: 'var(--text-primary)' }}>{fmt(m.cv_score, 'pct')}</div>
                                                </div>
                                            </>
                                        ) : (
                                            <>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>R²</div>
                                                    <div style={{ fontSize: '1.1rem', fontWeight: 900, color: isBest ? 'var(--accent-light)' : 'var(--text-primary)' }}>{fmt(m.r2 || m.score)}</div>
                                                </div>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>RMSE</div>
                                                    <div style={{ fontSize: '1.1rem', fontWeight: 900, color: 'var(--text-primary)' }}>{fmt(m.rmse)}</div>
                                                </div>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>MAE</div>
                                                    <div style={{ fontSize: '1.1rem', fontWeight: 900, color: 'var(--text-primary)' }}>{fmt(m.mae)}</div>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                    <ChevronDown size={16} color="var(--text-muted)" style={{ transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                                </div>

                                {/* Expanded details */}
                                {isOpen && (
                                    <div style={{ marginTop: '1.25rem', paddingTop: '1.25rem', borderTop: '1px solid var(--glass-border)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '1rem' }}
                                        onClick={e => e.stopPropagation()}>
                                        {(isClassification
                                            ? [
                                                { k: 'Accuracy', v: m.accuracy || m.score },
                                                { k: 'Precision', v: m.precision },
                                                { k: 'Recall', v: m.recall },
                                                { k: 'F1 Score', v: m.f1_score },
                                                { k: 'AUC-ROC', v: m.auc_roc },
                                                { k: 'CV Mean', v: m.cv_score },
                                            ]
                                            : [
                                                { k: 'R²', v: m.r2 || m.score },
                                                { k: 'Adj. R²', v: m.adjusted_r2 },
                                                { k: 'RMSE', v: m.rmse },
                                                { k: 'MAE', v: m.mae },
                                                { k: 'MSE', v: m.mse },
                                                { k: 'CV Mean', v: m.cv_score },
                                            ]
                                        ).map(metric => (
                                            <div key={metric.k} style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '10px', padding: '0.9rem', textAlign: 'center' }}>
                                                <div style={{ fontSize: '0.58rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.3rem' }}>{metric.k}</div>
                                                <div style={{ fontSize: '1rem', fontWeight: 900, color: 'var(--text-primary)' }}>{fmt(metric.v, isClassification ? 'pct' : 'num')}</div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        );
                    })}

                    {/* Score comparison chart */}
                    {modelCards.length > 1 && (
                        <div style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid var(--glass-border)', borderRadius: '16px', padding: '1.5rem' }}>
                            <div style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1.5rem' }}>
                                Score Comparison
                            </div>
                            <div style={{ height: Math.max(250, modelCards.length * 32 + 30) }}>
                                <ResponsiveContainer>
                                    <BarChart
                                        data={modelCards.map(m => ({ name: m.name, score: Math.round((m.score || 0) * 10000) / 100 }))}
                                        layout="vertical" margin={{ left: 10, right: 30, top: 5, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.02)" horizontal={false} />
                                        <XAxis type="number" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
                                        <YAxis dataKey="name" type="category" stroke="#475569" fontSize={11} fontWeight={700} axisLine={false} tickLine={false} width={140} />
                                        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(255,255,255,0.02)' }} formatter={(v) => [`${v}%`, 'Score']} />
                                        <Bar dataKey="score" radius={[0, 6, 6, 0]} barSize={16}>
                                            {modelCards.map((_, i) => (
                                                <Cell key={i} fill={i === 0 ? '#34d399' : i === 1 ? '#10b981' : 'rgba(52,211,153,0.3)'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ══════════  TAB: VISUALS  ══════════ */}
            {activeTab === 'visuals' && (
                <div className="fade-in">
                    {availablePlots.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-muted)' }}>No visualisations available for this run.</div>
                    ) : (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '1.5rem' }}>
                            {availablePlots.map(plot => (
                                <div key={plot.name} style={{
                                    background: 'rgba(255,255,255,0.015)',
                                    border: '1px solid var(--glass-border)',
                                    borderRadius: '16px',
                                    overflow: 'hidden',
                                    transition: 'border-color 0.2s'
                                }}>
                                    <div style={{ padding: '1rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                                        <span style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--text-primary)', textTransform: 'capitalize' }}>{plot.name}</span>
                                        <a href={plot.url} target="_blank" rel="noreferrer"
                                            style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--accent-light)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                            <ExternalLink size={12} /> Full Size
                                        </a>
                                    </div>
                                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.5rem' }}>
                                        <img src={plot.url} alt={plot.name} loading="lazy" style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '8px' }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* ══════════  TAB: EXPORT  ══════════ */}
            {activeTab === 'export' && (
                <div className="fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                    {/* Download ZIP */}
                    <div style={{
                        background: 'rgba(255,255,255,0.015)',
                        border: '1px solid var(--glass-border)',
                        borderRadius: '16px',
                        padding: '2.5rem',
                        display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', gap: '1.25rem'
                    }}>
                        <div style={{ width: '56px', height: '56px', borderRadius: '16px', background: 'var(--accent-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <FileDown size={28} color="var(--accent-light)" />
                        </div>
                        <div>
                            <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.3rem' }}>Full Project Bundle</div>
                            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Code, reports, plots, and pipeline trace</div>
                        </div>
                        <button className="btn btn-accent" style={{ padding: '0.85rem 2rem', fontSize: '0.9rem', borderRadius: '12px' }} onClick={handleDownloadZip}>
                            <Download size={16} /> Download ZIP
                        </button>
                    </div>

                    {/* Python Code */}
                    <div style={{
                        background: 'rgba(255,255,255,0.015)',
                        border: '1px solid var(--glass-border)',
                        borderRadius: '16px',
                        display: 'flex', flexDirection: 'column', overflow: 'hidden'
                    }}>
                        <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.03)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <Code2 size={14} color="var(--accent-light)" />
                                <span style={{ fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>analysis.py</span>
                            </div>
                            <CopyBtn text={generatedCode || ''} />
                        </div>
                        <pre style={{
                            padding: '1.25rem',
                            margin: 0,
                            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                            fontSize: '0.78rem',
                            color: '#94a3b8',
                            lineHeight: 1.7,
                            overflow: 'auto',
                            maxHeight: '400px',
                            background: 'rgba(0,0,0,0.25)'
                        }}>
                            {generatedCode?.slice(0, 3000) || '# No code generated'}
                            {generatedCode?.length > 3000 && '\n\n// ... truncated. Download ZIP for full file.'}
                        </pre>
                    </div>
                </div>
            )}
        </div>
    );
}
