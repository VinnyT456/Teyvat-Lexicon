import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { Check, ChevronRight, Copy, Eraser, Languages, LoaderCircle, Mic, RefreshCw, Settings2, Sparkles, Square, Star, Volume2, Wifi, WifiOff, X } from 'lucide-react';

const API = 'http://127.0.0.1:8000/api/v1';
const targets = [
  { code: 'en', name: 'English' }, { code: 'ja', name: 'Japanese' },
  { code: 'zhCN', name: 'Simplified Chinese' }, { code: 'zhTW', name: 'Traditional Chinese' },
] as const;
type Target = typeof targets[number]['code'];
type SourceLanguage = 'auto' | 'ru';
type TranslationSize = 'standard' | 'large';
type TranscriptionState = 'idle' | 'recording' | 'transcribing';
type Term = { id: string; en: string; ja: string; zhCN?: string | null; zhTW?: string | null };
type Ranked = { rank: number; match_type: 'exact' | 'semantic'; chroma_distance: number | null; term: Term };
type Result = { original: string; target_language: Target; translation: string; retrieved_terms: Ranked[] };
type ServiceHealth = {
  status: string;
  model_server_reachable: boolean;
  translation_model_loaded: boolean;
  translation_model_state: string;
  translation_model_url: string;
  translation_model_name: string;
  dictionary_ready: boolean;
  dictionary_entries: number;
  semantic_model_loaded: boolean;
  speech_model_loaded: boolean;
  transcription_model_loaded: boolean;
  transcription_model_state: string;
};
type Notice = { kind: 'error' | 'success'; message: string } | null;
type SpeechState = 'idle' | 'preparing' | 'playing';
type DetectedLanguage = { code: 'en' | 'ja' | 'cjk' | 'ru'; name: string };

function detectSourceLanguage(text: string): DetectedLanguage | null {
  const letters = Array.from(text).filter((character) => /\p{L}/u.test(character));
  if (letters.length < 2) return null;
  const count = (pattern: RegExp) => letters.filter((character) => pattern.test(character)).length;
  const kana = count(/[\u3040-\u30ff\u31f0-\u31ff]/u);
  const han = count(/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/u);
  const cyrillic = count(/[\u0400-\u052f]/u);
  const latin = count(/\p{Script=Latin}/u);
  if (kana >= 2 || (kana > 0 && kana + han >= letters.length * 0.16)) return { code: 'ja', name: 'Japanese' };
  if (cyrillic > 0 && cyrillic >= latin * 0.35) return { code: 'ru', name: 'Russian' };
  if (han > 0) return { code: 'cjk', name: 'Chinese / Japanese characters' };
  if (latin >= 2) return { code: 'en', name: 'English' };
  return null;
}

function readFavoriteTerms(): Term[] {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem('teyvat.favorite-terms') ?? '[]');
    if (!Array.isArray(value)) return [];
    return value.filter((item): item is Term => item && typeof item.id === 'string' && typeof item.en === 'string' && typeof item.ja === 'string')
      .slice(0, 100)
      .map((term) => ({ ...term, zhCN: typeof term.zhCN === 'string' ? term.zhCN : null, zhTW: typeof term.zhTW === 'string' ? term.zhTW : null }));
  } catch { return []; }
}

const QUICK_MODE = new URLSearchParams(window.location.search).get('mode') === 'quick';
const SETTINGS_MODE = new URLSearchParams(window.location.search).get('mode') === 'settings';
const MAX_RECORDING_MS = 30_000;

function getDetail(error: unknown) { return error instanceof Error ? error.message : 'Something went wrong. Please try again.'; }

function readPreference<T extends string>(key: string, choices: readonly T[], fallback: T): T {
  try {
    const saved = window.localStorage.getItem(`teyvat.${key}`);
    return choices.includes(saved as T) ? saved as T : fallback;
  } catch { return fallback; }
}

function useServiceHealth() {
  const [online, setOnline] = useState<boolean | null>(null);
  const [health, setHealth] = useState<ServiceHealth | null>(null);
  const [checkingConnection, setCheckingConnection] = useState(false);

  const checkHealth = useCallback(async (manual = false) => {
    if (manual) setCheckingConnection(true);
    try {
      const response = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2500) });
      if (!response.ok) throw new Error('Health check failed');
      setHealth(await response.json() as ServiceHealth);
      setOnline(true);
    } catch {
      setHealth(null);
      setOnline(false);
    } finally {
      if (manual) setCheckingConnection(false);
    }
  }, []);

  useEffect(() => {
    void checkHealth();
    const timer = window.setInterval(() => void checkHealth(), 12000);
    return () => window.clearInterval(timer);
  }, [checkHealth]);

  return { online, health, checkingConnection, checkHealth };
}

function connectionSummary(online: boolean | null, health: ServiceHealth | null) {
  if (online === null) return 'Checking FastAPI and the local model server.';
  if (online === false) return 'FastAPI is offline. Start the local service, then check again.';
  if (!health?.model_server_reachable) return 'FastAPI is online, but llama.cpp could not be reached.';
  if (!health.dictionary_ready) return 'The model server responds, but the glossary is not ready.';
  if (health.translation_model_loaded) return `${health.dictionary_entries.toLocaleString()} glossary entries ready.`;
  if (health.translation_model_state === 'sleeping') return 'Connected. The model is asleep and should wake when you translate.';
  return `Connected, but the model reports ${health.translation_model_state.replace(/_/g, ' ')}.`;
}

function ConnectionCheckButton({ compact = false, checking, onCheck }: { compact?: boolean; checking: boolean; onCheck: () => void }) {
  return <button className={`connection-check-button${compact ? ' compact' : ''}`} type="button" onClick={onCheck} disabled={checking} aria-label="Check local API and model connection" title="Check local API and model connection">
    {checking ? <LoaderCircle className="spin" size={14}/> : <RefreshCw size={14}/>}
    {!compact && <span>{checking ? 'Checking…' : 'Check connection'}</span>}
  </button>;
}

type SettingsPanelProps = {
  sourceLanguage: SourceLanguage;
  target: Target;
  translationSize: TranslationSize;
  online: boolean | null;
  health: ServiceHealth | null;
  checkingConnection: boolean;
  onCheckConnection: () => void;
  onSourceLanguageChange: (value: SourceLanguage) => void;
  onTargetChange: (value: Target) => void;
  onTranslationSizeChange: (value: TranslationSize) => void;
  onClose: () => void;
  onReset: () => void;
};

function SettingsPanel({ sourceLanguage, target, translationSize, online, health, checkingConnection, onCheckConnection, onSourceLanguageChange, onTargetChange, onTranslationSizeChange, onClose, onReset }: SettingsPanelProps) {
  const connectionState = online === false || (online === true && !health?.model_server_reachable)
    ? 'offline'
    : online === true && health?.translation_model_loaded ? 'ready' : '';

  return <>
    <div className="settings-head"><div><p className="settings-kicker">MAKE IT YOURS</p><h2 id="settings-title">Settings</h2></div><button className="settings-close" type="button" onClick={onClose} aria-label="Close settings"><X size={18}/></button></div>
    <section className="settings-section" aria-labelledby="defaults-title">
      <h3 id="defaults-title">Translation defaults</h3>
      <label className="setting-row" htmlFor="setting-source"><span><strong>Source language</strong><small>Auto-detects text; Cyrillic uses Russian prompts</small></span><select id="setting-source" value={sourceLanguage} onChange={(event) => onSourceLanguageChange(event.target.value as SourceLanguage)}><option value="auto">Auto-detect</option><option value="ru">Russian</option></select></label>
      <label className="setting-row" htmlFor="setting-target"><span><strong>Target language</strong><small>Default for new translations</small></span><select id="setting-target" value={target} onChange={(event) => onTargetChange(event.target.value as Target)}>{targets.map((item) => <option key={item.code} value={item.code}>{item.name}</option>)}</select></label>
    </section>
    <section className="settings-section" aria-labelledby="reading-title">
      <h3 id="reading-title">Reading</h3>
      <label className="setting-row" htmlFor="setting-size"><span><strong>Translation text</strong><small>Adjust the size for easier reading</small></span><select id="setting-size" value={translationSize} onChange={(event) => onTranslationSizeChange(event.target.value as TranslationSize)}><option value="standard">Standard</option><option value="large">Large</option></select></label>
    </section>
    <section className="settings-section" aria-labelledby="connection-title">
      <h3 id="connection-title">Connection</h3>
      <div className={`settings-connection-status ${connectionState}`} role="status" aria-live="polite">
        <span className="connection-dot" aria-hidden="true" />
        <div><strong>{online === null ? 'Checking local services' : online === false ? 'FastAPI offline' : health?.translation_model_loaded ? 'Ready to translate' : health?.model_server_reachable ? 'Model server connected' : 'Model server offline'}</strong><small>{connectionSummary(online, health)}</small></div>
        <ConnectionCheckButton compact checking={checkingConnection} onCheck={onCheckConnection}/>
      </div>
    </section>
    <section className="settings-section settings-about" aria-label="App information">
      <h3>Quick tips & privacy</h3>
      <p><kbd>⌘</kbd>/<kbd>Ctrl</kbd> + <kbd>↵</kbd> translates. Add <kbd>Shift</kbd> + <kbd>C</kbd>, <kbd>L</kbd>, or <kbd>M</kbd> to copy, listen, or speak a new line. <kbd>⌘</kbd>/<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Space</kbd> opens Quick Translate.</p>
      <p>Language and display preferences plus up to 100 favorite glossary terms are saved on this device. Source text, translations, and translation history are never saved.</p>
    </section>
    <div className="settings-footer"><button className="reset-button" type="button" onClick={onReset}>Reset defaults</button><span>TEYVAT LEXICON</span></div>
  </>;
}

function SettingsWindow() {
  const [sourceLanguage, setSourceLanguage] = useState<SourceLanguage>(() => readPreference('source-language', ['auto', 'ru'], 'auto'));
  const [target, setTarget] = useState<Target>(() => readPreference('target-language', targets.map((item) => item.code), 'en'));
  const [translationSize, setTranslationSize] = useState<TranslationSize>(() => readPreference('translation-size', ['standard', 'large'], 'standard'));
  const { online, health, checkingConnection, checkHealth } = useServiceHealth();

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === 'teyvat.source-language') {
        if (event.newValue === 'auto' || event.newValue === 'ru') setSourceLanguage(event.newValue);
      } else if (event.key === 'teyvat.target-language') {
        if (targets.some((item) => item.code === event.newValue)) setTarget(event.newValue as Target);
      } else if (event.key === 'teyvat.translation-size') {
        if (event.newValue === 'standard' || event.newValue === 'large') setTranslationSize(event.newValue);
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const savePreference = (key: string, value: string) => {
    try { window.localStorage.setItem(`teyvat.${key}`, value); } catch { /* Keep the setting usable for this window. */ }
  };
  const updateSourceLanguage = (value: SourceLanguage) => { setSourceLanguage(value); savePreference('source-language', value); };
  const updateTarget = (value: Target) => { setTarget(value); savePreference('target-language', value); };
  const updateTranslationSize = (value: TranslationSize) => { setTranslationSize(value); savePreference('translation-size', value); };
  const closeSettings = () => { if (isTauri()) void invoke('hide_settings_window_command'); else window.close(); };

  return <main className="settings-window-shell"><div className="settings-window-panel">
    <SettingsPanel
      sourceLanguage={sourceLanguage}
      target={target}
      translationSize={translationSize}
      online={online}
      health={health}
      checkingConnection={checkingConnection}
      onCheckConnection={() => void checkHealth(true)}
      onSourceLanguageChange={updateSourceLanguage}
      onTargetChange={updateTarget}
      onTranslationSizeChange={updateTranslationSize}
      onClose={closeSettings}
      onReset={() => { updateSourceLanguage('auto'); updateTarget('en'); updateTranslationSize('standard'); }}
    />
  </div></main>;
}

export default function App() {
  return SETTINGS_MODE ? <SettingsWindow /> : <TranslationApp />;
}

function TranslationApp() {
  const [source, setSource] = useState('');
  const [sourceLanguage, setSourceLanguage] = useState<SourceLanguage>(() => readPreference('source-language', ['auto', 'ru'], 'auto'));
  const [target, setTarget] = useState<Target>(() => readPreference('target-language', targets.map((item) => item.code), 'en'));
  const [translationSize, setTranslationSize] = useState<TranslationSize>(() => readPreference('translation-size', ['standard', 'large'], 'standard'));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [favoriteTerms, setFavoriteTerms] = useState<Term[]>(readFavoriteTerms);
  const [expandedTermId, setExpandedTermId] = useState<string | null>(null);
  const [pronunciation, setPronunciation] = useState<{ text: string; target: Target; value: string; label: string } | null>(null);
  const [pronunciationOpen, setPronunciationOpen] = useState(false);
  const [pronunciationBusy, setPronunciationBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const { online, health, checkingConnection, checkHealth } = useServiceHealth();
  const [notice, setNotice] = useState<Notice>(null);
  const [copied, setCopied] = useState(false);
  const [endpointCopied, setEndpointCopied] = useState(false);
  const [speechState, setSpeechState] = useState<SpeechState>('idle');
  const [transcriptionState, setTranscriptionState] = useState<TranscriptionState>('idle');
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const sourceRef = useRef<HTMLTextAreaElement>(null);
  const settingsRef = useRef<HTMLDialogElement>(null);
  const streamRef = useRef<AbortController | null>(null);
  const speechRef = useRef<AbortController | null>(null);
  const pronunciationRef = useRef<AbortController | null>(null);
  const transcriptionRef = useRef<AbortController | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<number | null>(null);
  const discardRecordingRef = useRef(false);
  const recordingRunRef = useRef(0);
  const speechRunRef = useRef(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const translation = result?.translation ?? '';
  const selectedName = targets.find((item) => item.code === target)?.name ?? 'English';
  const detectedLanguage = detectSourceLanguage(source);
  const sourceHint = sourceLanguage === 'ru'
    ? 'Russian selected'
    : detectedLanguage
      ? detectedLanguage.code === 'ru' ? 'Cyrillic detected · Russian mode' : `${detectedLanguage.name} detected`
      : 'One snippet, up to 20,000 characters';

  const stopSpeech = useCallback((updateState = true) => {
    speechRunRef.current += 1;
    speechRef.current?.abort();
    speechRef.current = null;
    if (audioSourceRef.current) {
      const source = audioSourceRef.current;
      source.onended = null;
      try { source.stop(); } catch { /* The source may already have ended. */ }
      source.disconnect();
      audioSourceRef.current = null;
    }
    if (updateState) setSpeechState('idle');
  }, []);

  const cancelSpeechInput = useCallback(() => {
    recordingRunRef.current += 1;
    discardRecordingRef.current = true;
    if (recordingTimerRef.current !== null) window.clearTimeout(recordingTimerRef.current);
    recordingTimerRef.current = null;
    transcriptionRef.current?.abort();
    transcriptionRef.current = null;
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    recorderRef.current = null;
    microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
    microphoneStreamRef.current = null;
    audioChunksRef.current = [];
    setTranscriptionState('idle');
    setRecordingSeconds(0);
  }, []);

  const cancelPronunciation = useCallback(() => {
    pronunciationRef.current?.abort();
    pronunciationRef.current = null;
    setPronunciation(null);
    setPronunciationOpen(false);
    setPronunciationBusy(false);
  }, []);

  const selectSourceLanguage = useCallback((value: SourceLanguage) => {
    streamRef.current?.abort(); cancelPronunciation(); cancelSpeechInput(); stopSpeech(); setSourceLanguage(value); setResult(null); setNotice(null);
  }, [cancelPronunciation, cancelSpeechInput, stopSpeech]);
  const selectTarget = useCallback((value: Target) => {
    streamRef.current?.abort(); cancelPronunciation(); cancelSpeechInput(); stopSpeech(); setTarget(value); setResult(null); setNotice(null);
  }, [cancelPronunciation, cancelSpeechInput, stopSpeech]);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === 'teyvat.source-language') {
        if (event.newValue === 'auto' || event.newValue === 'ru') selectSourceLanguage(event.newValue);
      } else if (event.key === 'teyvat.target-language') {
        if (targets.some((item) => item.code === event.newValue)) selectTarget(event.newValue as Target);
      } else if (event.key === 'teyvat.translation-size') {
        if (event.newValue === 'standard' || event.newValue === 'large') setTranslationSize(event.newValue);
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [selectSourceLanguage, selectTarget]);

  useEffect(() => { try { window.localStorage.setItem('teyvat.source-language', sourceLanguage); } catch { /* Preferences remain usable if storage is unavailable. */ } }, [sourceLanguage]);
  useEffect(() => { try { window.localStorage.setItem('teyvat.target-language', target); } catch { /* Preferences remain usable if storage is unavailable. */ } }, [target]);
  useEffect(() => { try { window.localStorage.setItem('teyvat.translation-size', translationSize); } catch { /* Preferences remain usable if storage is unavailable. */ } }, [translationSize]);
  useEffect(() => { try { window.localStorage.setItem('teyvat.favorite-terms', JSON.stringify(favoriteTerms)); } catch { setNotice({ kind: 'error', message: 'Favorites could not be saved on this device. Free some app storage and try again.' }); } }, [favoriteTerms]);
  useEffect(() => {
    const dialog = settingsRef.current;
    if (!dialog) return;
    if (settingsOpen && !dialog.open) dialog.showModal();
    else if (!settingsOpen && dialog.open) dialog.close();
  }, [settingsOpen]);

  useEffect(() => () => {
    stopSpeech(false);
    if (recordingTimerRef.current !== null) window.clearTimeout(recordingTimerRef.current);
    if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
    microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
    transcriptionRef.current?.abort();
    pronunciationRef.current?.abort();
    void audioContextRef.current?.close();
    audioContextRef.current = null;
  }, [stopSpeech]);

  useEffect(() => {
    if (transcriptionState !== 'recording') return;
    const timer = window.setInterval(() => setRecordingSeconds((seconds) => seconds + 1), 1000);
    return () => window.clearInterval(timer);
  }, [transcriptionState]);

  useEffect(() => {
    if (!QUICK_MODE) return;
    const focusSource = () => sourceRef.current?.focus();
    focusSource();
    if (!isTauri()) return;
    let unlisten: (() => void) | undefined;
    let unlistenHidden: (() => void) | undefined;
    let disposed = false;
    void listen('quick-window-focused', focusSource).then((stopListening) => {
      if (disposed) stopListening();
      else unlisten = stopListening;
    });
    void listen('quick-window-hidden', cancelSpeechInput).then((stopListening) => {
      if (disposed) stopListening();
      else unlistenHidden = stopListening;
    });
    return () => { disposed = true; unlisten?.(); unlistenHidden?.(); };
  }, [cancelSpeechInput]);

  useEffect(() => {
    if (!QUICK_MODE) return;
    document.documentElement.classList.add('quick-popover');
    document.body.classList.add('quick-popover');
    return () => {
      document.documentElement.classList.remove('quick-popover');
      document.body.classList.remove('quick-popover');
    };
  }, []);

  const translate = useCallback(async (input?: string) => {
    if (busy) return;
    const text = input ?? source;
    cancelSpeechInput();
    if (!text.trim()) { sourceRef.current?.focus(); setNotice({ kind: 'error', message: 'Add a line of text to translate.' }); return; }
    stopSpeech();
    cancelPronunciation();
    const sourceLanguageHint = sourceLanguage === 'ru' || detectSourceLanguage(text)?.code === 'ru' ? 'ru' : 'auto';
    const controller = new AbortController(); streamRef.current = controller;
    setBusy(true); setResult(null); setNotice(null); setCopied(false);
    try {
      const response = await fetch(`${API}/translations/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' }, body: JSON.stringify({ text, target_language: target, source_language: sourceLanguageHint }), signal: controller.signal });
      if (response.status === 404 || response.status === 405) {
        const fallback = await fetch(`${API}/translations`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, target_language: target, source_language: sourceLanguageHint }), signal: controller.signal });
        const payload = await fallback.json().catch(() => ({}));
        if (!fallback.ok) {
          const detail = typeof payload.detail === 'string' ? payload.detail : fallback.status === 503 ? 'The local service is busy. Try again shortly.' : 'The translation service could not complete this request.';
          throw new Error(detail);
        }
        const current = sourceRef.current?.value === text && document.querySelector<HTMLInputElement>('input[name="target-language"]:checked')?.value === target;
        if (current) { setResult(payload as Result); setNotice({ kind: 'success', message: `Translation ready in ${selectedName}.` }); void checkHealth(); }
        return;
      }
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = typeof payload.detail === 'string' ? payload.detail : response.status === 503 ? 'The local service is busy. Try again shortly.' : 'The translation service could not complete this request.';
        throw new Error(detail);
      }
      if (!response.body) throw new Error('The translation stream is unavailable. Please try again.');
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = ''; let receivedDone = false;
      const isCurrent = () => sourceRef.current?.value === text && document.querySelector<HTMLInputElement>('input[name="target-language"]:checked')?.value === target;
      const consume = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line) as { type?: string; retrieved_terms?: Ranked[]; text?: string; translation?: string; detail?: string };
        if (!isCurrent()) { controller.abort(); return; }
        if (event.type === 'terms') setResult((current) => ({ original: text, target_language: target, translation: current?.translation ?? '', retrieved_terms: event.retrieved_terms ?? [] }));
        else if (event.type === 'token') setResult((current) => ({ original: text, target_language: target, translation: `${current?.translation ?? ''}${event.text ?? ''}`, retrieved_terms: current?.retrieved_terms ?? [] }));
        else if (event.type === 'done') { receivedDone = true; setResult((current) => ({ original: text, target_language: target, translation: event.translation ?? current?.translation ?? '', retrieved_terms: current?.retrieved_terms ?? [] })); setNotice({ kind: 'success', message: `Translation ready in ${selectedName}.` }); void checkHealth(); }
        else if (event.type === 'error') throw new Error(event.detail || 'The translation service could not complete this request.');
      };
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const lines = pending.split('\n'); pending = lines.pop() ?? '';
        for (const line of lines) consume(line.replace(/\r$/, ''));
        if (done) break;
      }
      if (pending.trim()) consume(pending);
      if (!controller.signal.aborted && isCurrent() && !receivedDone) throw new Error('The translation stream ended before it was complete. Please try again.');
    } catch (error) {
      if (controller.signal.aborted) return;
      setResult(null);
      const message = error instanceof TypeError ? 'Cannot reach the local translation service. Start it and try again.' : getDetail(error);
      setNotice({ kind: 'error', message }); void checkHealth();
    } finally { if (streamRef.current === controller) streamRef.current = null; setBusy(false); }
  }, [busy, cancelPronunciation, cancelSpeechInput, checkHealth, selectedName, source, sourceLanguage, stopSpeech, target]);

  const recordSpeechInput = useCallback(async () => {
    if (transcriptionState === 'recording') {
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
      return;
    }
    if (transcriptionState === 'transcribing' || busy) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setNotice({ kind: 'error', message: 'This app window cannot record audio on this device.' });
      return;
    }

    const run = ++recordingRunRef.current;
    discardRecordingRef.current = false;
    audioChunksRef.current = [];
    setNotice(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (run !== recordingRunRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const mimeType = [
        'audio/webm;codecs=opus',
        'audio/mp4',
        'audio/ogg;codecs=opus',
        'audio/webm',
      ].find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      recorderRef.current = recorder;
      microphoneStreamRef.current = stream;
      recorder.ondataavailable = (event) => { if (event.data.size > 0) audioChunksRef.current.push(event.data); };
      recorder.onstop = async () => {
        if (recordingTimerRef.current !== null) window.clearTimeout(recordingTimerRef.current);
        recordingTimerRef.current = null;
        stream.getTracks().forEach((track) => track.stop());
        if (recorderRef.current === recorder) recorderRef.current = null;
        if (microphoneStreamRef.current === stream) microphoneStreamRef.current = null;

        const audio = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        audioChunksRef.current = [];
        if (discardRecordingRef.current || run !== recordingRunRef.current) {
          discardRecordingRef.current = false;
          return;
        }

        setTranscriptionState('transcribing');
        const controller = new AbortController();
        transcriptionRef.current = controller;
        try {
          const response = await fetch(`${API}/transcriptions?source_language=${sourceLanguage}`, {
            method: 'POST',
            headers: { 'Content-Type': audio.type || 'audio/webm' },
            body: audio,
            signal: controller.signal,
          });
          const payload = await response.json().catch(() => ({})) as { text?: string; detail?: string };
          if (!response.ok) {
            throw new Error(payload.detail || (response.status === 503 ? 'The local speech recognizer is busy. Try again shortly.' : 'The speech could not be transcribed. Try again.'));
          }
          const recognized = payload.text?.trim();
          if (!recognized) throw new Error('I couldn’t hear clear speech. Try speaking again.');

          if (sourceRef.current) sourceRef.current.value = recognized;
          setSource(recognized);
          setResult(null);
          setNotice({ kind: 'success', message: 'Speech recognized. Translating…' });
          await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
          if (!controller.signal.aborted && run === recordingRunRef.current) await translate(recognized);
        } catch (error) {
          if (!controller.signal.aborted) {
            const message = error instanceof TypeError
              ? 'Cannot reach the local speech service. Start FastAPI and try again.'
              : getDetail(error);
            setNotice({ kind: 'error', message });
          }
        } finally {
          if (transcriptionRef.current === controller) transcriptionRef.current = null;
          if (run === recordingRunRef.current) setTranscriptionState('idle');
          setRecordingSeconds(0);
        }
      };

      recorder.start();
      setTranscriptionState('recording');
      setRecordingSeconds(0);
      recordingTimerRef.current = window.setTimeout(() => {
        if (recorder.state === 'recording') recorder.stop();
      }, MAX_RECORDING_MS);
    } catch (error) {
      if (run !== recordingRunRef.current) return;
      const message = error instanceof DOMException && error.name === 'NotAllowedError'
        ? 'Microphone access is off. Allow it for Teyvat Lexicon, then try again.'
        : error instanceof TypeError
          ? 'This app window cannot record audio on this device.'
          : getDetail(error);
      setTranscriptionState('idle');
      setNotice({ kind: 'error', message });
    }
  }, [busy, sourceLanguage, transcriptionState, translate]);

  const playTranslation = useCallback(async () => {
    if (speechState === 'playing' || speechState === 'preparing') { stopSpeech(); return; }
    if (!translation) return;

    stopSpeech();
    const run = speechRunRef.current;
    const controller = new AbortController();
    speechRef.current = controller;
    setNotice(null);
    setSpeechState('preparing');
    try {
      // Resume synchronously from the click gesture; WebKit may reject audio
      // playback if activation is lost while the local model is generating it.
      const audioContext = audioContextRef.current ?? new AudioContext();
      audioContextRef.current = audioContext;
      if (audioContext.state !== 'running') await audioContext.resume();

      const response = await fetch(`${API}/speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: translation, target_language: target }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = typeof payload.detail === 'string' ? payload.detail : '';
        if (response.status === 404 || response.status === 405) {
          throw new Error('Restart FastAPI with the latest code and run uv sync to enable local voice playback.');
        }
        throw new Error(detail || (response.status === 503 ? 'The local voice is busy. Try again shortly.' : 'The local voice could not speak this translation.'));
      }

      const audioData = await response.arrayBuffer();
      const decodedAudio = await audioContext.decodeAudioData(audioData);
      if (controller.signal.aborted || speechRunRef.current !== run) {
        return;
      }
      const source = audioContext.createBufferSource();
      source.buffer = decodedAudio;
      source.connect(audioContext.destination);
      source.onended = () => {
        if (speechRunRef.current !== run) return;
        audioSourceRef.current = null;
        stopSpeech();
      };
      audioSourceRef.current = source;
      source.start();
      if (speechRunRef.current === run) setSpeechState('playing');
    } catch (error) {
      if (controller.signal.aborted || speechRunRef.current !== run) return;
      const message = error instanceof TypeError
        ? 'Cannot reach the local voice service. Start FastAPI and try again.'
        : error instanceof DOMException
          ? error.name === 'NotAllowedError'
            ? 'Audio was blocked by the app window. Tap LISTEN to enable playback, then try again.'
            : `Audio playback failed (${error.name}). Try the translation again.`
          : getDetail(error);
      stopSpeech();
      setNotice({ kind: 'error', message });
    } finally {
      if (speechRef.current === controller) speechRef.current = null;
    }
  }, [speechState, stopSpeech, target, translation]);

  const togglePronunciation = useCallback(async () => {
    if (!translation || target === 'en') return;
    if (pronunciation?.text === translation && pronunciation.target === target) {
      setPronunciationOpen((open) => !open);
      return;
    }
    pronunciationRef.current?.abort();
    const controller = new AbortController(); pronunciationRef.current = controller;
    const text = translation; const language = target;
    setPronunciationBusy(true); setNotice(null);
    try {
      const response = await fetch(`${API}/pronunciations`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, target_language: language }), signal: controller.signal });
      const payload = await response.json().catch(() => ({})) as { pronunciation?: string; label?: string; detail?: string };
      if (!response.ok) throw new Error(payload.detail || 'Pronunciation could not be generated. Try again.');
      if (controller.signal.aborted) return;
      if (!payload.pronunciation?.trim()) throw new Error('No pronunciation was returned. Try again.');
      setPronunciation({ text, target: language, value: payload.pronunciation, label: payload.label ?? (language === 'ja' ? 'romaji' : 'pinyin') });
      setPronunciationOpen(true);
    } catch (error) {
      if (controller.signal.aborted) return;
      setNotice({ kind: 'error', message: error instanceof TypeError ? 'Cannot reach the local pronunciation service. Start FastAPI and try again.' : getDetail(error) });
    } finally {
      if (pronunciationRef.current === controller) { pronunciationRef.current = null; setPronunciationBusy(false); }
    }
  }, [pronunciation, target, translation]);

  const copyResult = useCallback(async () => {
    if (!translation) return;
    try { await navigator.clipboard.writeText(translation); setCopied(true); setNotice({ kind: 'success', message: 'Translation copied to clipboard.' }); window.setTimeout(() => setCopied(false), 1800); }
    catch { setNotice({ kind: 'error', message: 'Clipboard access is unavailable.' }); }
  }, [translation]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const modified = event.metaKey || event.ctrlKey;
      if (modified && !event.shiftKey && event.key === 'Enter') { event.preventDefault(); void translate(); return; }
      if (!modified || !event.shiftKey) return;
      const key = event.key.toLowerCase();
      if (key === 'c' && translation) { event.preventDefault(); void copyResult(); }
      else if (key === 'l' && translation) { event.preventDefault(); void playTranslation(); }
      else if (key === 'm' && !busy) { event.preventDefault(); void recordSpeechInput(); }
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [busy, copyResult, playTranslation, recordSpeechInput, translate, translation]);

  function clearCurrentLine() {
    streamRef.current?.abort();
    cancelPronunciation();
    cancelSpeechInput();
    stopSpeech();
    if (sourceRef.current) sourceRef.current.value = '';
    setSource(''); setResult(null); setCopied(false); setNotice(null); setExpandedTermId(null);
    window.requestAnimationFrame(() => sourceRef.current?.focus());
  }

  function toggleFavorite(term: Term) {
    const exists = favoriteTerms.some((favorite) => favorite.id === term.id);
    if (!exists && favoriteTerms.length >= 100) { setNotice({ kind: 'error', message: 'You can save up to 100 glossary terms. Remove one to make room.' }); return; }
    setFavoriteTerms((current) => exists ? current.filter((favorite) => favorite.id !== term.id) : [...current, term]);
    setNotice({ kind: 'success', message: exists ? 'Removed from favorite terms.' : 'Saved to favorite terms.' });
  }

  const favoriteIds = new Set(favoriteTerms.map((term) => term.id));

  async function copyModelEndpoint() {
    try {
      await navigator.clipboard.writeText(health?.translation_model_url ?? 'http://127.0.0.1:9931/v1');
      setEndpointCopied(true);
      window.setTimeout(() => setEndpointCopied(false), 1800);
    } catch { setNotice({ kind: 'error', message: 'Clipboard access is unavailable.' }); }
  }

  const modelState = health?.translation_model_state;
  const serviceLabel = online === false
    ? 'API offline'
    : online === null
      ? 'Checking API'
      : !health
        ? 'Checking model'
        : health.translation_model_loaded
          ? 'Model ready'
          : modelState === 'loading'
            ? 'Model loading'
            : modelState === 'sleeping'
              ? 'Model asleep'
              : modelState === 'auth_required'
                ? 'Model auth needed'
                : modelState === 'not_loaded'
                  ? 'Model not loaded'
                  : modelState === 'server_error'
                    ? 'Model server error'
                    : modelState === 'offline'
                      ? 'Model server offline'
                      : 'Model status unknown';
  const modelServerOffline = online === true && health?.model_server_reachable === false;
  const serviceTone = online === false || modelServerOffline
    ? 'offline'
    : online === true && health?.status !== 'ready'
      ? 'degraded'
      : '';
  const serviceDetails = health
    ? `FastAPI online · ${health.translation_model_name}: ${health.translation_model_state} · ${health.dictionary_ready ? `${health.dictionary_entries} glossary entries ready` : 'glossary unavailable'} · semantic model ${health.semantic_model_loaded ? 'loaded' : 'loads when needed'} · speech model ${health.speech_model_loaded ? 'loaded' : 'loads on first play'} · transcription model ${health.transcription_model_state ?? 'loads on first use'}`
    : online === false ? 'FastAPI is offline' : 'Checking FastAPI and model status';
  const modelServerAddress = (() => {
    try { return new URL(health?.translation_model_url ?? 'http://127.0.0.1:9931/v1').host; }
    catch { return 'Local model server'; }
  })();
  const modelDisplayName = health?.translation_model_name.split('/').pop() ?? 'Local GGUF model';
  const modelStatusLabel = online === false
    ? 'Local API offline'
    : !health
      ? 'Checking model status'
      : !health.model_server_reachable
        ? 'Model server offline'
        : health.translation_model_loaded
          ? 'Loaded and ready'
          : health.translation_model_state === 'loading'
            ? 'Loading model…'
            : health.translation_model_state === 'sleeping'
              ? 'Asleep · wakes on request'
              : health.translation_model_state.replace(/_/g, ' ');

  return <main className={`shell${QUICK_MODE ? ' quick-shell' : ''}`}>
    <header className={`top${QUICK_MODE ? ' quick-top' : ''}`}>
      <div className="brand-mark" aria-hidden="true"><svg viewBox="0 0 40 40" role="presentation"><path className="book-page" d="M7 11.5c5.4-2.2 9.7-1.3 13 1.4v19c-3.3-2.7-7.6-3.6-13-1.4z"/><path className="book-page" d="M33 11.5c-5.4-2.2-9.7-1.3-13 1.4v19c3.3-2.7 7.6-3.6 13-1.4z"/><path className="book-line" d="M10.5 17c3-.8 5.4-.3 7.5 1.1M10.5 21c3-.8 5.4-.3 7.5 1.1M29.5 17c-3-.8-5.4-.3-7.5 1.1M29.5 21c-3-.8-5.4-.3-7.5 1.1"/><path className="book-star" d="m28 4 1.4 3.5L33 9l-3.6 1.4L28 14l-1.4-3.6L23 9l3.6-1.5z"/></svg></div>
      <div className="brand" data-tauri-drag-region>Teyvat Lexicon<small>{QUICK_MODE ? 'QUICK TRANSLATE' : 'GENSHIN TRANSLATION DESK'}</small></div>
      {QUICK_MODE ? <div className="quick-window-controls">
        <div className={`service quick-service ${serviceTone}`} role="status" aria-live="polite" aria-label={serviceDetails} title={serviceDetails}>{online === false || modelServerOffline ? <WifiOff size={14}/> : modelState === 'loading' ? <LoaderCircle className="spin" size={14}/> : <Wifi size={14}/>}<span>{serviceLabel}</span></div>
        <ConnectionCheckButton compact checking={checkingConnection} onCheck={() => void checkHealth(true)}/>
        <button className="quick-close-button" type="button" onClick={() => void invoke('hide_quick_window_command')} aria-label="Hide quick translator"><X size={17}/></button>
      </div> : <div className="header-actions">
        <div className={`service ${serviceTone}`} role="status" aria-live="polite" aria-label={serviceDetails} title={serviceDetails}>
          {online === false || modelServerOffline ? <WifiOff size={15}/> : modelState === 'loading' ? <LoaderCircle className="spin" size={15}/> : <Wifi size={15}/>}<span>{serviceLabel}</span>
        </div>
        <ConnectionCheckButton checking={checkingConnection} onCheck={() => void checkHealth(true)}/>
        <button className="settings-button" type="button" onClick={() => setSettingsOpen(true)} aria-haspopup="dialog" aria-label="Open settings"><Settings2 size={16}/><span>SETTINGS</span></button>
      </div>}
    </header>

    {QUICK_MODE && <section className="popover-engine" aria-label="Local model status">
      <div className="popover-server-row">
        <span className={`popover-online-dot${online === false || modelServerOffline ? ' is-offline' : health?.translation_model_loaded ? ' is-ready' : ''}`} aria-hidden="true" />
        <div className="popover-server-info"><strong>Local model server</strong><span>{modelServerAddress}</span></div>
        <button className="popover-copy" type="button" onClick={() => void copyModelEndpoint()} aria-label={endpointCopied ? 'Model server address copied' : 'Copy model server address'} title={endpointCopied ? 'Copied' : 'Copy address'}>{endpointCopied ? <Check size={15}/> : <Copy size={15}/>}</button>
        <button className="popover-link" type="button" onClick={() => void invoke('show_main_window_command')}>Open desk</button>
      </div>
      <div className="popover-divider" />
      <p className="popover-section-title">INSTALLED MODEL</p>
      <div className="popover-model-row">
        <span className="popover-model-mark" aria-hidden="true"><Sparkles size={20}/></span>
        <span className="popover-model-info"><strong title={health?.translation_model_name}>{modelDisplayName}</strong><small>{modelStatusLabel}</small></span>
        <ChevronRight className="popover-model-chevron" size={18}/>
      </div>
    </section>}

    <section className="heading">
      <div><h1>{QUICK_MODE ? 'Quick Translate' : 'Character Index'}</h1><p className="subhead">{QUICK_MODE ? 'Speak or paste a game snippet.' : 'Translate the line. Check its recognized names below.'}</p></div>
      <fieldset className="target"><legend>TARGET LANGUAGE</legend><div className="language-options">{targets.map((item) => <label className="language-choice" key={item.code}><input type="radio" name="target-language" value={item.code} checked={target === item.code} onChange={() => selectTarget(item.code)} /><span>{item.name}</span></label>)}</div></fieldset>
    </section>

    <section className="work" aria-label="Translation">
      <div className="pane source-pane">
        <div className="caption source-caption"><label htmlFor="source-text">SOURCE TEXT</label><div className="source-controls"><label className="source-language-picker" htmlFor="source-language"><span>LANGUAGE</span><select id="source-language" value={sourceLanguage} onChange={(event) => selectSourceLanguage(event.target.value as SourceLanguage)}><option value="auto">Auto-detect</option><option value="ru">Russian</option></select></label>{(source || result) && <button className="clear-source" type="button" onClick={clearCurrentLine} aria-label="Clear text and translation" title="Start a new line"><Eraser size={14}/></button>}<button className={`record-button${transcriptionState === 'recording' ? ' is-recording' : ''}`} type="button" onClick={() => void recordSpeechInput()} disabled={busy || transcriptionState === 'transcribing'} aria-label={transcriptionState === 'recording' ? 'Stop recording and translate' : transcriptionState === 'transcribing' ? 'Transcribing speech' : 'Speak to translate'} aria-busy={transcriptionState === 'transcribing'}>{transcriptionState === 'recording' ? <Square size={13} fill="currentColor"/> : transcriptionState === 'transcribing' ? <LoaderCircle className="spin" size={14}/> : <Mic size={14}/>}<span>{transcriptionState === 'recording' ? 'STOP & TRANSLATE' : transcriptionState === 'transcribing' ? 'TRANSCRIBING' : 'SPEAK'}</span></button></div></div>
        <textarea id="source-text" ref={sourceRef} value={source} maxLength={20000} onChange={(event) => { streamRef.current?.abort(); cancelPronunciation(); if (transcriptionState !== 'idle') cancelSpeechInput(); stopSpeech(); setSource(event.target.value); setResult(null); setNotice(null); }} placeholder="Paste or speak your Genshin Impact text…" aria-describedby="source-hint" />
        <div className="pane-foot"><span id="source-hint">{transcriptionState === 'recording' ? `Listening… ${recordingSeconds}s / 30s` : transcriptionState === 'transcribing' ? 'Transcribing locally… first use downloads the model' : sourceHint}</span><span>{source.length.toLocaleString()} / 20,000</span></div>
      </div>
      <div className={`pane output-pane${pronunciationOpen ? ' with-pronunciation' : ''}`}>
        <div className="caption"><span id="translation-label">TRANSLATION</span><span className="output-language">{selectedName.toUpperCase()}</span></div>
        {busy && !translation ? <div className="output-state loading"><LoaderCircle className="spin" size={20}/><span>Finding the right words…</span></div> : translation ? <div className="translation-group" aria-labelledby="translation-label" aria-live={busy ? 'off' : 'polite'}><div className={`translation${busy ? ' streaming' : ''}${translationSize === 'large' ? ' large' : ''}`}>{translation}{busy && <span className="typing-cursor" aria-hidden="true"/>}</div>{pronunciationOpen && pronunciation && <div className="pronunciation-line"><span>{pronunciation.label.toUpperCase()}</span>{pronunciation.value}</div>}</div> : <div className="output-state">Your translation will appear here</div>}
        {translation && !busy && <div className="output-tools">
          {speechState !== 'idle' && <span className="speech-status" role="status">{speechState === 'preparing' ? 'Preparing local voice… first play downloads it once' : 'Speaking the translation'}</span>}
          <button className={`speech-button${speechState === 'playing' ? ' is-playing' : ''}`} type="button" onClick={() => void playTranslation()} aria-label={speechState === 'preparing' ? 'Cancel voice preparation' : speechState === 'playing' ? 'Stop speaking' : 'Play translation aloud'} aria-busy={speechState === 'preparing'}>{speechState === 'preparing' ? <X size={15}/> : speechState === 'playing' ? <Square size={14} fill="currentColor"/> : <Volume2 size={16}/>}<span>{speechState === 'preparing' ? 'CANCEL' : speechState === 'playing' ? 'STOP' : 'LISTEN'}</span></button>
          {target !== 'en' && <button className={`pronunciation-button${pronunciationOpen ? ' is-active' : ''}`} type="button" onClick={() => void togglePronunciation()} disabled={pronunciationBusy} aria-label={pronunciationBusy ? 'Loading pronunciation' : pronunciationOpen ? 'Hide pronunciation' : `Show ${target === 'ja' ? 'romaji' : 'pinyin'}`} aria-busy={pronunciationBusy}>{pronunciationBusy ? <LoaderCircle className="spin" size={14}/> : <Languages size={14}/>}<span>{pronunciationBusy ? '…' : target === 'ja' ? 'ROMAJI' : 'PINYIN'}</span></button>}
          <button className="copy-button" type="button" onClick={copyResult} aria-label={copied ? 'Translation copied' : 'Copy translation'}>{copied ? <Check size={15}/> : <Copy size={15}/>}<span>{copied ? 'COPIED' : 'COPY'}</span></button>
        </div>}
      </div>
      <div className="actions"><button className="action" type="button" onClick={() => void translate()} disabled={busy}><span>{busy ? 'Translating…' : 'Translate text'}</span>{busy ? <LoaderCircle className="spin" size={16}/> : <Sparkles size={16}/>}</button><span className="shortcut"><kbd>⌘</kbd><span>/</span><kbd>Ctrl</kbd><kbd>↵</kbd></span></div>
    </section>

    <section className="glossary" aria-labelledby="glossary-title">
      <div className="ghead"><div><h2 id="glossary-title">Ranked glossary terms</h2><p>{result ? `${result.retrieved_terms.length} ${result.retrieved_terms.length === 1 ? 'match' : 'matches'} · highest rank first` : 'Recognized names and terms from this line'}</p></div></div>
      {result?.retrieved_terms.length ? <div className="list">{result.retrieved_terms.map((item) => { const localized = target === 'ja' ? item.term.ja : target === 'zhCN' ? item.term.zhCN : target === 'zhTW' ? item.term.zhTW : item.term.en; const primary = localized?.trim() || item.term.en; const reference = primary === item.term.en ? item.term.ja : item.term.en; const expanded = expandedTermId === item.term.id; const favorite = favoriteIds.has(item.term.id); return <article className={`row${expanded ? ' is-expanded' : ''}`} key={`${item.rank}-${item.term.id}`} style={{ animationDelay: `${Math.min(item.rank - 1, 5) * 35}ms` }}>
        <div className="number">{String(item.rank).padStart(2, '0')}</div>
        <button className="termname" type="button" onClick={() => setExpandedTermId(expanded ? null : item.term.id)} aria-expanded={expanded} aria-label={`${expanded ? 'Hide' : 'Show'} details for ${primary}`}><span>{primary}</span><small>{reference}</small></button>
        <div className="match"><span className={`badge ${item.match_type}`}>{item.match_type === 'exact' ? 'EXACT MATCH' : 'SEMANTIC MATCH'}</span>{item.match_type === 'semantic' && item.chroma_distance !== null && item.chroma_distance !== undefined && <small className="distance" title="Raw ChromaDB distance. Lower is closer; this is not a probability.">distance {item.chroma_distance.toFixed(3)} <span>· lower is closer</span></small>}</div>
        <button className={`term-favorite${favorite ? ' is-saved' : ''}`} type="button" onClick={() => toggleFavorite(item.term)} aria-label={favorite ? `Remove ${primary} from favorite terms` : `Save ${primary} to favorite terms`} title={favorite ? 'Remove favorite' : 'Save favorite'}><Star size={17} fill={favorite ? 'currentColor' : 'none'}/></button>
        {expanded && <div className="term-details"><div className="term-translations">{targets.map((language) => <div key={language.code}><small>{language.name}</small><span>{language.code === 'en' ? item.term.en : language.code === 'ja' ? item.term.ja : language.code === 'zhCN' ? item.term.zhCN || '—' : item.term.zhTW || '—'}</span></div>)}</div><p>{item.match_type === 'exact' ? 'This phrase appears directly in the source line.' : 'This term was retrieved by semantic similarity.'}{item.match_type === 'semantic' && item.chroma_distance !== null && item.chroma_distance !== undefined ? ` Raw distance: ${item.chroma_distance.toFixed(3)}; lower is closer, not a probability.` : ''}</p></div>}
      </article>;})}</div> : <div className="glossary-empty">{result ? 'No glossary terms matched this line.' : 'Translate a line to reveal its ranked names and terms.'}</div>}
      <p className="foot">Exact phrase matches rank first. Semantic distances are raw scores; lower is closer, not a probability.</p>
    </section>

    {!QUICK_MODE && <section className="favorites" aria-labelledby="favorites-title">
      <div className="ghead"><div><h2 id="favorites-title">Favorite glossary terms</h2><p>{favoriteTerms.length} saved on this device · source snippets are never saved</p></div></div>
      {favoriteTerms.length ? <div className="favorite-list">{favoriteTerms.map((term) => { const localized = target === 'ja' ? term.ja : target === 'zhCN' ? term.zhCN : target === 'zhTW' ? term.zhTW : term.en; const primary = localized?.trim() || term.en; return <article className="favorite-term" key={term.id}><Star size={15} fill="currentColor" aria-hidden="true"/><div><strong>{primary}</strong><small>{term.en === primary ? term.ja : term.en}</small></div><button type="button" onClick={() => toggleFavorite(term)} aria-label={`Remove ${primary} from favorite terms`} title="Remove favorite"><X size={15}/></button></article>;})}</div> : <div className="favorites-empty">Tap the star beside a ranked term to keep it here for later.</div>}
    </section>}

    <div className="sr-status" role="status" aria-live="polite">{notice?.kind === 'success' ? notice.message : ''}</div>
    {notice?.kind === 'error' && <div className="notice error" role="alert">{notice.message}</div>}
    {notice?.kind === 'success' && <div className="notice success" aria-hidden="true">{notice.message}</div>}
    <footer className="bottom"><span>LOCAL TRANSLATION · NO SAVED HISTORY</span><span>FOUR TARGET LANGUAGES</span></footer>

    {QUICK_MODE && <footer className="popover-footer">
      <span className="popover-footer-note">LOCAL · NO HISTORY</span>
      <div className="popover-footer-actions">
        <button type="button" onClick={() => void invoke('open_settings_command')}><Settings2 size={15}/>Settings</button>
        <button type="button" onClick={() => void invoke('quit_application_command')}>Quit</button>
      </div>
    </footer>}

    <dialog ref={settingsRef} className="settings-dialog" aria-labelledby="settings-title" onClose={() => setSettingsOpen(false)} onClick={(event) => { if (event.target === settingsRef.current) setSettingsOpen(false); }}>
      <SettingsPanel
        sourceLanguage={sourceLanguage}
        target={target}
        translationSize={translationSize}
        online={online}
        health={health}
        checkingConnection={checkingConnection}
        onCheckConnection={() => void checkHealth(true)}
        onSourceLanguageChange={selectSourceLanguage}
        onTargetChange={selectTarget}
        onTranslationSizeChange={setTranslationSize}
        onClose={() => setSettingsOpen(false)}
        onReset={() => { selectSourceLanguage('auto'); selectTarget('en'); setTranslationSize('standard'); }}
      />
    </dialog>
  </main>;
}
