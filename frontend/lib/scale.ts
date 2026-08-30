// Measured fleet scale-test data — the decisive concurrency probe: Qwen3-30B-A3B FP8 on a 4-GPU
// tier (TP=4), single relevant-cart serve path vs BM25-chunked RAG with a CHURNED prefix cache
// (the multi-tenant regime), server-measured TTFT/latency, GPU priced at $4.602/hr, measured
// 1..24 concurrent queries. Shared by the Scale Test tab AND the Costs tab so the fleet cost story
// comes from one source of truth instead of a single-query estimate. A production build would
// stream these from a live scale-test run; the shape here matches what that endpoint would return.

export const GPU_HOURLY = 4.602;
export const SCALE_MAX = 24; // measured ceiling on the 4-GPU serving tier

export type Arm = { qps: number; ttft: number; lat: number };
export type ScalePoint = { u: number; cart: Arm; rag: Arm };

export const SCALE_PTS: ScalePoint[] = [
  { u: 1, cart: { qps: 1.78, ttft: 28, lat: 586 }, rag: { qps: 1.36, ttft: 276, lat: 744 } },
  { u: 4, cart: { qps: 3.92, ttft: 47, lat: 947 }, rag: { qps: 2.04, ttft: 296, lat: 1873 } },
  { u: 8, cart: { qps: 5.25, ttft: 61, lat: 1361 }, rag: { qps: 2.41, ttft: 388, lat: 2991 } },
  { u: 16, cart: { qps: 7.55, ttft: 80, lat: 2027 }, rag: { qps: 2.73, ttft: 399, lat: 5405 } },
  { u: 24, cart: { qps: 9.22, ttft: 91, lat: 2284 }, rag: { qps: 2.98, ttft: 648, lat: 7677 } },
];

// GPU $/query at a sustained throughput (queries/sec): hourly ÷ (qps × 3600). At scale this is far
// below the single-query cost, because continuous batching amortizes the GPU across many in-flight
// queries — the realistic fleet number.
export const costPerQuery = (qps: number) => GPU_HOURLY / (qps * 3600);

const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

// Interpolate arm metrics at a continuous concurrency u in [1, SCALE_MAX].
export function scaleAt(u: number): { u: number; cart: Arm; rag: Arm } {
  if (u <= SCALE_PTS[0].u) return { u, cart: SCALE_PTS[0].cart, rag: SCALE_PTS[0].rag };
  const last = SCALE_PTS[SCALE_PTS.length - 1];
  if (u >= last.u) return { u: last.u, cart: last.cart, rag: last.rag };
  for (let i = 1; i < SCALE_PTS.length; i++) {
    if (u <= SCALE_PTS[i].u) {
      const a = SCALE_PTS[i - 1], b = SCALE_PTS[i], f = (u - a.u) / (b.u - a.u);
      const mix = (x: Arm, y: Arm): Arm => ({
        qps: lerp(x.qps, y.qps, f), ttft: lerp(x.ttft, y.ttft, f), lat: lerp(x.lat, y.lat, f),
      });
      return { u, cart: mix(a.cart, b.cart), rag: mix(a.rag, b.rag) };
    }
  }
  return { u: last.u, cart: last.cart, rag: last.rag };
}

// The realistic fleet operating point = saturation (max measured concurrency).
export const atScale = SCALE_PTS[SCALE_PTS.length - 1];
