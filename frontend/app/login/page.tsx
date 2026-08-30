"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
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

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("register");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [tenant, setTenant] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [googleEnabled, setGoogleEnabled] = useState(false);

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

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const res =
        mode === "register"
          ? await api.register(email, password, tenant)
          : await api.login(email, password);
      setToken(res.access_token);
      router.push("/");
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
          <h1 className="text-xl font-semibold">Cartridge KV Platform</h1>
          <p className="text-sm text-slate-500">
            {mode === "register" ? "Create your tenant account" : "Sign in to your tenant"}
          </p>
        </CardHeader>
        <CardBody>
          <form onSubmit={submit} className="space-y-3">
            {mode === "register" && (
              <div>
                <Label>Organization</Label>
                <Input
                  data-testid="tenant-input"
                  value={tenant}
                  onChange={(e: any) => setTenant(e.target.value)}
                  placeholder="Acme Inc"
                  required
                />
              </div>
            )}
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
              <Label>Password</Label>
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
              {busy ? "Please wait…" : mode === "register" ? "Create account" : "Sign in"}
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

          <div className="mt-4 flex items-center justify-between">
            <button
              data-testid="toggle-mode"
              className="text-sm text-slate-400 hover:text-slate-200"
              onClick={() => setMode(mode === "register" ? "login" : "register")}
            >
              {mode === "register" ? "Have an account? Sign in" : "Need an account? Register"}
            </button>
            <div className="flex items-center gap-4 text-sm">
              <a href="/about" data-testid="about-public-link" className="text-slate-400 hover:text-slate-200">
                About
              </a>
              <a href="/demo" data-testid="demo-public-link" className="text-emerald-400 hover:underline">
                Cost comparison →
              </a>
            </div>
          </div>
        </CardBody>
      </Card>
    </main>
  );
}
