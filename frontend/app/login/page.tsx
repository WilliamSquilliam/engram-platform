"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, setToken } from "@/lib/api";
import { Button, Input, Label, Card, CardBody, CardHeader } from "@/components/ui";

const ERROR_LABELS: Record<string, string> = {
  google_auth_failed: "Google sign-in failed. Please try again.",
  google_email_unverified: "Your Google account email is not verified.",
};

// Official Google "G" mark, inline so there's no extra asset to ship.
function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62z" />
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18z" />
      <path fill="#FBBC05" d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33z" />
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58A8.99 8.99 0 0 0 9 0 9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z" />
    </svg>
  );
}

// The app is invite-only: you either sign in, or ask to be added to the waitlist. There is no
// open self-serve registration — new tenants are approved manually and emailed an invite.
export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "request">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [googleEnabled, setGoogleEnabled] = useState(false);

  // Request-access panel state.
  const [reqName, setReqName] = useState("");
  const [reqTenant, setReqTenant] = useState("");
  const [reqReason, setReqReason] = useState("");
  const [requested, setRequested] = useState(false);

  // On mount: (1) finish a Google round-trip if we were redirected back with a
  // token in the URL fragment, (2) surface any OAuth error from the query string,
  // (3) ask the backend whether Google is configured so we only show the button
  // when it actually works.
  useEffect(() => {
    const t = new URLSearchParams(window.location.hash.slice(1)).get("token");
    if (t) {
      setToken(t);
      router.push("/");
      return;
    }
    const err = new URLSearchParams(window.location.search).get("error");
    if (err) setError(ERROR_LABELS[err] || "Sign-in failed.");
    api
      .authConfig()
      .then((c) => setGoogleEnabled(!!c.google_enabled))
      .catch(() => {});
  }, []);

  async function submitLogin(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const res = await api.login(email, password);
      setToken(res.access_token);
      router.push("/");
    } catch (err: any) {
      setError(err.message || "Failed");
    } finally {
      setBusy(false);
    }
  }

  async function submitRequest(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await api.requestAccess({
        email,
        name: reqName,
        tenant_name: reqTenant,
        ...(reqReason ? { reason: reqReason } : {}),
      });
      setRequested(true);
    } catch (err: any) {
      setError(err.message || "Failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <h1 className="text-xl font-semibold">Engram Smart CAG</h1>
          <p className="text-sm text-slate-500">
            {mode === "login" ? "Sign in" : "Request access (invite-only)"}
          </p>
        </CardHeader>
        <CardBody>
          {mode === "login" ? (
            <>
              <form onSubmit={submitLogin} className="space-y-3">
                <div>
                  <Label>Email</Label>
                  <Input
                    data-testid="email-input"
                    type="email"
                    value={email}
                    onChange={(e: any) => setEmail(e.target.value)}
                    placeholder="admin@acme.test"
                    required
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between">
                    <Label>Password</Label>
                    <Link
                      href="/forgot-password"
                      data-testid="forgot-password-link"
                      className="mb-1 text-xs text-slate-400 hover:text-slate-200"
                    >
                      Forgot password?
                    </Link>
                  </div>
                  <Input
                    data-testid="password-input"
                    type="password"
                    value={password}
                    onChange={(e: any) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    required
                  />
                </div>
                {error && (
                  <p data-testid="auth-error" className="text-sm text-red-400">
                    {error}
                  </p>
                )}
                <Button data-testid="submit-btn" type="submit" className="w-full" disabled={busy}>
                  {busy ? "Please wait…" : "Sign in"}
                </Button>
              </form>

              {googleEnabled && (
                <>
                  <div className="my-4 flex items-center gap-3 text-xs text-slate-500">
                    <span className="h-px flex-1 bg-slate-700" />
                    or
                    <span className="h-px flex-1 bg-slate-700" />
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    data-testid="google-signin"
                    className="w-full gap-2"
                    onClick={() => {
                      window.location.href = api.googleLoginUrl();
                    }}
                  >
                    <GoogleIcon />
                    Continue with Google
                  </Button>
                </>
              )}
            </>
          ) : requested ? (
            // Waitlist confirmation — no auto sign-in; a human approves and emails an invite.
            <div data-testid="request-confirmation" className="space-y-3 py-2 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-300">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <h2 className="font-medium text-slate-100">You&apos;re on the waitlist</h2>
              <p className="text-sm text-slate-400">
                Thanks for your interest. We&apos;ll review your request and email you an invite at{" "}
                <span className="text-slate-200">{email}</span> when your account is ready.
              </p>
            </div>
          ) : (
            <form onSubmit={submitRequest} className="space-y-3">
              <div>
                <Label>Work email</Label>
                <Input
                  data-testid="request-email"
                  type="email"
                  value={email}
                  onChange={(e: any) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  required
                />
              </div>
              <div>
                <Label>Your name</Label>
                <Input
                  data-testid="request-name"
                  value={reqName}
                  onChange={(e: any) => setReqName(e.target.value)}
                  placeholder="Jane Doe"
                  required
                />
              </div>
              <div>
                <Label>Organization</Label>
                <Input
                  data-testid="request-tenant"
                  value={reqTenant}
                  onChange={(e: any) => setReqTenant(e.target.value)}
                  placeholder="Acme Inc"
                  required
                />
              </div>
              <div>
                <Label>Describe your company and needs (optional)</Label>
                <textarea
                  data-testid="request-reason"
                  value={reqReason}
                  onChange={(e) => setReqReason(e.target.value)}
                  placeholder="A sentence on your use case helps us prioritize."
                  rows={3}
                  className="w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:ring-2 focus:ring-slate-600"
                />
              </div>
              {error && (
                <p data-testid="auth-error" className="text-sm text-red-400">
                  {error}
                </p>
              )}
              <Button data-testid="request-submit" type="submit" className="w-full" disabled={busy}>
                {busy ? "Please wait…" : "Request access"}
              </Button>
              <p className="text-center text-xs text-slate-500">
                Signup is invite-only. We&apos;ll email you once you&apos;re approved.
              </p>
            </form>
          )}

          <div className="mt-4 flex items-center justify-between">
            <button
              data-testid="toggle-mode"
              className="text-sm text-slate-400 hover:text-slate-200"
              onClick={() => {
                setError("");
                setRequested(false);
                setMode(mode === "login" ? "request" : "login");
              }}
            >
              {mode === "login" ? "Need access? Request an invite" : "Have an account? Sign in"}
            </button>
            <a href="/about" data-testid="about-public-link" className="text-sm text-slate-400 hover:text-slate-200">
              About
            </a>
          </div>
        </CardBody>
      </Card>
    </main>
  );
}
