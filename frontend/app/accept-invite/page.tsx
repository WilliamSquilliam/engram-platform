"use client";
// Accept-invite: a teammate lands here from an emailed link (/accept-invite#token=...), sets a
// password, and is dropped straight into the app with a fresh session token.
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, readUrlToken, setToken } from "@/lib/api";
import { Button, Input, Label, Card, CardBody, CardHeader } from "@/components/ui";

function AcceptInviteInner() {
  const router = useRouter();
  // The token rides the URL fragment (client-only), so resolve it in an effect:
  // null = still resolving, "" = genuinely missing.
  const [token, setInviteToken] = useState<string | null>(null);
  useEffect(() => {
    setInviteToken(readUrlToken() ?? "");
  }, []);

  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!token) return;
    setError("");
    if (password.length < 8) {
      setError("Use at least 8 characters.");
      return;
    }
    if (password !== confirm) {
      setError("Passwords don't match.");
      return;
    }
    setBusy(true);
    try {
      const res = await api.acceptInvite(token, password, name || undefined);
      // Fresh signup: keep them signed in on this device (persistent store; token
      // lifetime is the backend default — they can pick session-only next login).
      setToken(res.access_token, true);
      router.push("/");
    } catch (err: any) {
      setError(err.message || "Couldn't accept this invite.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <h1 className="text-xl font-semibold">Set your password</h1>
          <p className="text-sm text-slate-500">Finish joining your team on Engram.</p>
        </CardHeader>
        <CardBody>
          {token === null ? null : !token ? (
            <div className="space-y-3 text-sm text-slate-400">
              <p data-testid="invite-error">
                This invite link is missing its token. Please use the link from your email, or ask your
                admin to send a new invite.
              </p>
              <Link href="/login" className="text-slate-300 hover:text-slate-100">
                Back to sign in
              </Link>
            </div>
          ) : (
            <form onSubmit={submit} className="space-y-3">
              <div>
                <Label>Your name (optional)</Label>
                <Input
                  data-testid="invite-name"
                  value={name}
                  onChange={(e: any) => setName(e.target.value)}
                  placeholder="Jane Doe"
                />
              </div>
              <div>
                <Label>Password</Label>
                <Input
                  data-testid="invite-password"
                  type="password"
                  value={password}
                  onChange={(e: any) => setPassword(e.target.value)}
                  placeholder="At least 8 characters"
                  required
                />
              </div>
              <div>
                <Label>Confirm password</Label>
                <Input
                  data-testid="invite-confirm"
                  type="password"
                  value={confirm}
                  onChange={(e: any) => setConfirm(e.target.value)}
                  placeholder="••••••••"
                  required
                />
              </div>
              {error && (
                <p data-testid="invite-error" className="text-sm text-red-400">
                  {error}
                </p>
              )}
              <Button data-testid="invite-submit" type="submit" className="w-full" disabled={busy}>
                {busy ? "Please wait…" : "Set password & continue"}
              </Button>
            </form>
          )}
        </CardBody>
      </Card>
    </main>
  );
}

export default function AcceptInvitePage() {
  return <AcceptInviteInner />;
}
