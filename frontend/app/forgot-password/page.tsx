"use client";
// Forgot password: post the email and always show the same neutral confirmation, so the page
// never reveals whether an account exists for that address.
import { useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Button, Input, Label, Card, CardBody, CardHeader } from "@/components/ui";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await api.forgotPassword(email);
      setSent(true);
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <h1 className="text-xl font-semibold">Reset your password</h1>
          <p className="text-sm text-slate-500">We&apos;ll email you a link to set a new one.</p>
        </CardHeader>
        <CardBody>
          {sent ? (
            <div data-testid="forgot-confirmation" className="space-y-3 py-2 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-300">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <p className="text-sm text-slate-400">
                If an account exists for <span className="text-slate-200">{email}</span>, we&apos;ve sent a
                password reset link. Check your inbox.
              </p>
              <Link href="/login" className="inline-block text-sm text-slate-300 hover:text-slate-100">
                Back to sign in
              </Link>
            </div>
          ) : (
            <form onSubmit={submit} className="space-y-3">
              <div>
                <Label>Email</Label>
                <Input
                  data-testid="forgot-email"
                  type="email"
                  value={email}
                  onChange={(e: any) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  required
                />
              </div>
              {error && (
                <p data-testid="forgot-error" className="text-sm text-red-400">
                  {error}
                </p>
              )}
              <Button data-testid="forgot-submit" type="submit" className="w-full" disabled={busy}>
                {busy ? "Please wait…" : "Send reset link"}
              </Button>
              <div className="text-center">
                <Link href="/login" className="text-sm text-slate-400 hover:text-slate-200">
                  Back to sign in
                </Link>
              </div>
            </form>
          )}
        </CardBody>
      </Card>
    </main>
  );
}
