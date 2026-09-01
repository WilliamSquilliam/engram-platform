"use client";
// New Corpus wizard — Step 1: name. Creating the corpus here gives it an id, so
// the rest of the flow (upload + train) lives at /document-base/[id]/setup and is
// resumable from the dashboard even if the user navigates away mid-training.
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getToken } from "@/lib/api";
import { Button, Input, Card, CardBody, CardHeader, Stepper } from "@/components/ui";

export default function NewCorpusPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!getToken()) router.push("/login");
  }, [router]);

  async function next(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    setErr("");
    try {
      const c = await api.createCorpus(name.trim());
      router.replace(`/document-base/${c.id}/setup`);
    } catch (e: any) {
      setErr(e.message || "Could not create document base");
      setBusy(false);
    }
  }

  return (
    <main className="max-w-xl mx-auto p-6">
      <button
        onClick={() => router.push("/")}
        className="text-sm text-slate-400 hover:text-slate-200"
        data-testid="wizard-cancel"
      >
        ← Cancel
      </button>
      <h1 className="text-2xl font-semibold mt-2 mb-4">New Document Base</h1>
      <div className="mb-6">
        <Stepper steps={["Name", "Documents", "Model", "Review", "Onboard"]} current={0} />
      </div>

      <Card>
        <CardHeader>
          <h2 className="font-medium">Name Your Document Base</h2>
          <p className="text-xs text-slate-400">
            A document base is also called a knowledge base, it is everything you want the AI to know.
          </p>
        </CardHeader>
        <CardBody>
          <form onSubmit={next} className="space-y-3">
            <Input
              data-testid="corpus-name"
              autoFocus
              value={name}
              onChange={(e: any) => setName(e.target.value)}
              placeholder="e.g. Support KB"
            />
            {err && <p className="text-sm text-red-400">{err}</p>}
            <div className="flex justify-end">
              <Button data-testid="wizard-next" type="submit" disabled={!name.trim() || busy}>
                {busy ? "Creating…" : "Continue"}
              </Button>
            </div>
          </form>
        </CardBody>
      </Card>
    </main>
  );
}
