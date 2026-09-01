"use client";
// Reset password: land here from the emailed link (/reset-password#token=...), set a new password,
// then go sign in with it.
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, readUrlToken } from "@/lib/api";
import { Button, Input, Label, Card, CardBody, CardHeader } from "@/components/ui";

function ResetPasswordInner() {
  const router = useRouter();
  // The token rides the URL fragment (client-only), so resolve it in an effect:
  // null = still resolving, "" = genuinely missing.
  const [token, setResetToken] = useState<string | null>(null);
  useEffect(() => {
    setResetToken(readUrlToken() ?? "");
  }, []);

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

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
      await api.resetPassword(token, password);
      setDone(true);
    } catch (err: any) {
      setError(err.message || "This reset link is invalid or has expired.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <h1 className="text-xl font-semibold">Choose a new password</h1>
          <p className="text-sm text-slate-500">Set a new password for your account.</p>
        </CardHeader>
        <CardBody>
          {done ? (
            <div data-testid="reset-confirmation" className="space-y-3 py-2 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-300">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <p className="text-sm text-slate-400">Your password has been reset.</p>
              <Button
                data-testid="reset-goto-login"
                className="w-full"
                onClick={() => router.push("/login")}
              >
                Sign in
              </Button>
            </div>
          ) : token === null ? null : !token ? (
            <div className="space-y-3 text-sm text-slate-400">
              <p data-testid="reset-error">
                This reset link is missing its token. Request a new one from the sign-in page.
              </p>
              <Link href="/forgot-password" className="text-slate-300 hover:text-slate-100">
                Request a new link
              </Link>
            </div>
          ) : (
            <form onSubmit={submit} className="space-y-3">
              <div>
                <Label>New password</Label>
                <Input
                  data-testid="reset-password"
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
                  data-testid="reset-confirm"
                  type="password"
                  value={confirm}
                  onChange={(e: any) => setConfirm(e.target.value)}
                  placeholder="••••••••"
                  required
                />
              </div>
              {error && (
                <p data-testid="reset-error" className="text-sm text-red-400">
                  {error}
                </p>
              )}
              <Button data-testid="reset-submit" type="submit" className="w-full" disabled={busy}>
                {busy ? "Please wait…" : "Reset password"}
              </Button>
            </form>
          )}
        </CardBody>
      </Card>
    </main>
  );
}

export default function ResetPasswordPage() {
  return <ResetPasswordInner />;
}
