// Energy-based VAD chunker. Accumulates mic frames until a natural speech
// boundary is detected (trailing silence after voice activity) OR a hard cap
// is reached. This replaces clock-bounded chunking so each flushed buffer is
// a coherent utterance — which is what Pyannote's embedding model needs to
// produce a stable speaker centroid.
//
// Design:
//   - We analyse the stream in fixed 20ms sub-frames (consistent with webrtcvad).
//   - Per-frame feature is RMS in dBFS.
//   - Noise floor is an EMA that only tracks *quiet* frames, so speech doesn't
//     drag the floor up. Threshold = noise_floor + margin (dB above floor).
//   - Hangover: once voice is detected we require `hangoverFrames` consecutive
//     quiet frames before declaring silence — this bridges natural pauses inside
//     an utterance ("um, ..., I was saying").
//   - We buffer audio from voice onset until the silence trigger fires. If the
//     buffer exceeds maxChunkMs we flush with reason "cap" mid-utterance — the
//     next chunk will carry the tail.
//   - minChunkMs prevents pathological micro-flushes (e.g. a single cough).

export type FlushReason = "vad" | "cap" | "stop"

export type VadChunkerConfig = {
  sampleRate: number
  minChunkMs?: number        // default 1500 — don't emit chunks shorter than this
  maxChunkMs?: number        // default 8000 — hard cap, forces "cap" flush. Shorter = lower latency, more mid-utterance splits.
  silenceMs?: number         // default 400 — trailing silence to trigger "vad" flush
  frameMs?: number           // default 20 — analysis frame size
  energyThresholdDb?: number // default 8 — dB above adaptive noise floor required to register as voice
  initialNoiseFloorDb?: number // default -55
  noiseFloorAlpha?: number   // default 0.02 — EMA rate when updating noise floor from silent frames
}

export type OnChunkCallback = (buffer: Float32Array, reason: FlushReason) => void

const DEFAULTS = {
  minChunkMs: 1500,
  maxChunkMs: 8000,
  silenceMs: 400,
  frameMs: 20,
  energyThresholdDb: 8,
  initialNoiseFloorDb: -55,
  noiseFloorAlpha: 0.02,
}

export class VadChunker {
  private readonly frameSamples: number
  private readonly minChunkSamples: number
  private readonly maxChunkSamples: number
  private readonly silenceFramesRequired: number
  private readonly energyThresholdDb: number
  private readonly noiseFloorAlpha: number

  // Analysis state
  private noiseFloorDb: number
  private voiceActive = false
  private silentRun = 0  // consecutive silent frames since last voice frame

  // Carry-over samples when feed() length isn't a clean frame multiple.
  private carry: Float32Array = new Float32Array(0)

  // Accumulated utterance audio (waiting to be flushed).
  private buffered: Float32Array[] = []
  private bufferedSamples = 0
  // Small trailing pre-roll of silent frames kept in buffered to give the
  // transcriber natural onset/decay. We only keep up to hangoverFrames worth.

  private onChunk: OnChunkCallback | null = null

  constructor(config: VadChunkerConfig) {
    const c = { ...DEFAULTS, ...config }
    this.frameSamples = Math.max(1, Math.round((c.frameMs * c.sampleRate) / 1000))
    this.minChunkSamples = Math.round((c.minChunkMs * c.sampleRate) / 1000)
    this.maxChunkSamples = Math.round((c.maxChunkMs * c.sampleRate) / 1000)
    this.silenceFramesRequired = Math.max(1, Math.round(c.silenceMs / c.frameMs))
    this.energyThresholdDb = c.energyThresholdDb
    this.noiseFloorAlpha = c.noiseFloorAlpha
    this.noiseFloorDb = c.initialNoiseFloorDb
  }

  setOnChunk(cb: OnChunkCallback): void {
    this.onChunk = cb
  }

  /** Push new mic samples. Frames are analysed and flushed inline when boundaries fire. */
  feed(input: Float32Array): void {
    if (input.length === 0) return
    // Join carry-over + new input into one working buffer.
    let work: Float32Array
    if (this.carry.length === 0) {
      work = input
    } else {
      work = new Float32Array(this.carry.length + input.length)
      work.set(this.carry, 0)
      work.set(input, this.carry.length)
    }

    const total = work.length
    let i = 0
    while (i + this.frameSamples <= total) {
      const frame = work.subarray(i, i + this.frameSamples)
      this.processFrame(frame)
      i += this.frameSamples
    }
    // Stash the tail that didn't complete a frame.
    this.carry = work.slice(i)
  }

  /** Drain any buffered audio (called on stop). */
  flushNow(reason: FlushReason = "stop"): void {
    // Include carry so we don't drop the tail < one frame.
    if (this.carry.length > 0) {
      this.appendToBuffer(this.carry)
      this.carry = new Float32Array(0)
    }
    if (this.bufferedSamples === 0) return
    this.emit(reason)
  }

  /** Reset all internal state (use when restarting). */
  reset(): void {
    this.voiceActive = false
    this.silentRun = 0
    this.carry = new Float32Array(0)
    this.buffered = []
    this.bufferedSamples = 0
  }

  // ---- internals ----

  private processFrame(frame: Float32Array): void {
    const db = rmsDb(frame)
    const threshold = this.noiseFloorDb + this.energyThresholdDb
    const isVoice = db > threshold

    if (isVoice) {
      this.voiceActive = true
      this.silentRun = 0
      this.appendToBuffer(frame)
    } else {
      // Adapt noise floor during quiet periods (EMA toward observed db).
      if (Number.isFinite(db)) {
        this.noiseFloorDb =
          (1 - this.noiseFloorAlpha) * this.noiseFloorDb + this.noiseFloorAlpha * db
        // Clamp to a sensible range.
        if (this.noiseFloorDb < -90) this.noiseFloorDb = -90
        if (this.noiseFloorDb > -20) this.noiseFloorDb = -20
      }
      if (this.voiceActive) {
        // Still inside an utterance — keep the quiet frame so hangover pauses
        // stay in the buffer.
        this.silentRun += 1
        this.appendToBuffer(frame)
        if (this.silentRun >= this.silenceFramesRequired) {
          // End of utterance detected.
          if (this.bufferedSamples >= this.minChunkSamples) {
            this.emit("vad")
          }
          // else: wait for more voice; short blip, keep accumulating.
          this.voiceActive = false
          this.silentRun = 0
        }
      }
      // else: pure silence before any voice — discard frame (don't buffer).
    }

    // Cap trigger regardless of voice state.
    if (this.bufferedSamples >= this.maxChunkSamples) {
      this.emit("cap")
      // After a cap cut we stay in whatever voice state we were — the
      // next frames continue the utterance into a new chunk.
      this.silentRun = 0
    }
  }

  private appendToBuffer(frame: Float32Array): void {
    // Copy because Float32Array.subarray shares the underlying buffer.
    const copy = new Float32Array(frame.length)
    copy.set(frame)
    this.buffered.push(copy)
    this.bufferedSamples += copy.length
  }

  private emit(reason: FlushReason): void {
    if (this.bufferedSamples === 0) return
    const merged = new Float32Array(this.bufferedSamples)
    let off = 0
    for (const p of this.buffered) {
      merged.set(p, off)
      off += p.length
    }
    this.buffered = []
    this.bufferedSamples = 0
    try {
      this.onChunk?.(merged, reason)
    } catch (err) {
      console.warn("VadChunker onChunk threw", err)
    }
  }
}

function rmsDb(frame: Float32Array): number {
  let sum = 0
  for (let i = 0; i < frame.length; i++) {
    const s = frame[i]
    sum += s * s
  }
  const rms = Math.sqrt(sum / Math.max(1, frame.length))
  if (rms <= 1e-9) return -120
  return 20 * Math.log10(rms)
}
