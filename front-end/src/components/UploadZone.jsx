import { useState, useRef } from 'react';
import { UploadCloud, FileText, ArrowRight, BarChart2, Cpu, Code2, Globe, Sparkles } from 'lucide-react';

export default function UploadZone({ onFileSelect }) {
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [targetColumn, setTargetColumn] = useState('');
    const inputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
        else if (e.type === 'dragleave') setDragActive(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]?.name.endsWith('.csv')) {
            setSelectedFile(e.dataTransfer.files[0]);
        } else {
            alert('Please upload a valid CSV file.');
        }
    };

    const handleChange = (e) => {
        if (e.target.files?.[0]) setSelectedFile(e.target.files[0]);
    };

    const handleStart = () => {
        if (!selectedFile) return alert('Please select a CSV file first.');
        if (!targetColumn.trim()) return alert('Please enter the target column name.');
        onFileSelect(selectedFile, targetColumn.trim());
    };

    const features = [
        { icon: BarChart2, title: 'Smart Analysis', desc: 'Automatic EDA & visualization' },
        { icon: Cpu, title: 'Multi-Model', desc: 'Trains & compares ML models' },
        { icon: Code2, title: 'Code Export', desc: 'Production-ready Python code' },
    ];

    return (
        <div className="fade-in" style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '5rem', alignItems: 'center', minHeight: 'calc(100vh - 160px)', padding: '2rem 0' }}>

            {/* ── Left: Hero Text ────────────────── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                <div className="stagger-1">
                    <div className="premium-tag" style={{ marginBottom: '1.5rem' }}>
                        <Sparkles size={12} style={{ marginRight: '0.4rem' }} /> 7 AI Agents · Real-Time Pipeline
                    </div>
                    <h1 style={{ fontSize: '4rem', fontWeight: 900, letterSpacing: '-0.05em', lineHeight: 1, color: 'var(--text-primary)', marginBottom: '1.25rem' }}>
                        From raw CSV<br />to <span style={{ color: 'var(--accent-light)' }}>trained model</span><br />in minutes.
                    </h1>
                    <p style={{ fontSize: '1.1rem', lineHeight: 1.7, color: 'var(--text-secondary)', maxWidth: '480px', fontWeight: 500 }}>
                        Upload your dataset, pick a target column, and let the AutoEDA ecosystem handle the heavy lifting — from preprocessing to production-ready code.
                    </p>
                </div>

                {/* Feature pills */}
                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }} className="stagger-2">
                    {features.map(f => (
                        <div key={f.title} className="card" style={{ padding: '0.75rem 1.25rem', border: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <f.icon size={16} className="card-icon" />
                            <div>
                                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{f.title}</div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 500 }}>{f.desc}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* ── Right: Upload Card ─────────────── */}
            <div className="card stagger-3" style={{ padding: '3rem', maxWidth: '480px', justifySelf: 'end', width: '100%', border: '1px solid var(--accent-border)' }}>
                <div style={{ marginBottom: '2rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>Initialize Pipeline</h2>
                    <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Select your dataset to begin autonomous analysis.</p>
                </div>

                {/* Dropzone */}
                <div
                    style={{
                        border: dragActive ? '2px dashed var(--accent)' : '1px dashed var(--glass-border)',
                        borderRadius: 'var(--radius)',
                        padding: '3rem 2rem',
                        textAlign: 'center',
                        cursor: 'pointer',
                        transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                        background: dragActive ? 'var(--accent-glow)' : 'rgba(255,255,255,0.02)',
                        marginBottom: '1.5rem',
                        boxShadow: dragActive ? '0 0 20px var(--accent-glow)' : 'none'
                    }}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => !selectedFile && inputRef.current.click()}
                >
                    <input ref={inputRef} type="file" accept=".csv" onChange={handleChange} style={{ display: 'none' }} />

                    {!selectedFile ? (
                        <>
                            <div style={{ display: 'inline-flex', padding: '1rem', background: 'var(--accent-glow)', borderRadius: '50%', marginBottom: '1rem' }}>
                                <UploadCloud size={32} color="var(--accent-light)" />
                            </div>
                            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>Drop CSV here</div>
                            <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>or click to browse files</div>
                        </>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                            <FileText size={40} color="var(--accent-light)" />
                            <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text-primary)', marginTop: '0.5rem' }}>{selectedFile.name}</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{(selectedFile.size / 1024).toFixed(1)} KB</div>
                        </div>
                    )}
                </div>

                {/* Target + Run */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    <div>
                        <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>
                            Target Column
                        </label>
                        <input
                            type="text"
                            className="input"
                            style={{ padding: '0.85rem 1rem' }}
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            placeholder="e.g. Price, Target, Category"
                        />
                    </div>
                    <button
                        className="btn btn-accent"
                        style={{ width: '100%', justifyContent: 'center', padding: '1rem', fontSize: '0.95rem', borderRadius: 'var(--radius)' }}
                        onClick={handleStart}
                        disabled={!selectedFile}
                    >
                        Deploy Pipeline <ArrowRight size={18} style={{ marginLeft: '0.5rem' }} />
                    </button>
                </div>
            </div>
        </div>
    );
}
