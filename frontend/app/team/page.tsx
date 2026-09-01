"use client";
// Team management (tenant-admin only): list members with roles, invite a teammate (and copy the
// returned invite link since email may be gated off), revoke pending invites, change a member's
// role, and remove a member. Non-admins who land here are bounced home.
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getToken } from "@/lib/api";
import type { Invite, Member, MemberRole, User } from "@/lib/types";
import { Button, Input, Label, Card, CardBody, CardHeader, Badge, cn } from "@/components/ui";

const ROLE_LABEL: Record<string, string> = { admin: "Admin", member: "Member" };

// Copy-to-clipboard button with a brief "Copied" state. Falls back to a select-and-copy hint if
// the clipboard API is unavailable (e.g. non-secure origin).
function CopyButton({ value, testId }: { value: string; testId?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      type="button"
      variant="outline"
      data-testid={testId}
      className="shrink-0"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard blocked — the link stays visible for manual copy */
        }
      }}
    >
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

export default function TeamPage() {
  const router = useRouter();
  const [me, setMe] = useState<User | null>(null);
  const [members, setMembers] = useState<Member[]>([]);
  const [invites, setInvites] = useState<Invite[]>([]);
  const [loading, setLoading] = useState(true);
  const [gateChecked, setGateChecked] = useState(false);
  const [error, setError] = useState("");

  // Invite form.
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<MemberRole>("member");
  const [inviting, setInviting] = useState(false);
  const [newInvite, setNewInvite] =
    useState<{ email: string; role: MemberRole; link: string | null } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      // One call: /admin/members returns both the members and the tenant's pending invites.
      const data = await api.listMembers();
      setMembers(data.members);
      setInvites(data.invites);
      setError("");
    } catch (err: any) {
      setError(err.message || "Couldn't load your team.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Gate on admin role: fetch /auth/me, bounce non-admins, otherwise load the team.
  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    (async () => {
      try {
        const m = await api.me();
        setMe(m);
        if (m.role !== "admin") {
          router.replace("/");
          return;
        }
        setGateChecked(true);
        await load();
      } catch {
        router.push("/login");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function submitInvite(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setInviting(true);
    try {
      const res = await api.createInvite(inviteEmail, inviteRole);
      setNewInvite({ email: inviteEmail, role: inviteRole, link: res.invite_link ?? null });
      setInviteEmail("");
      setInviteRole("member");
      await load();
    } catch (err: any) {
      setError(err.message || "Couldn't create the invite.");
    } finally {
      setInviting(false);
    }
  }

  // Wait until the admin gate resolves so we never flash the team UI to a non-admin.
  if (!gateChecked) return <div className="p-8 text-slate-400">Loading…</div>;

  return (
    <div className="mx-auto max-w-4xl p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">Team</h1>
        <p className="text-sm text-slate-400">Invite teammates and manage their access.</p>
      </div>

      {error && (
        <p data-testid="team-error" className="mb-4 text-sm text-red-400">
          {error}
        </p>
      )}

      {/* Invite a teammate */}
      <Card className="mb-6">
        <CardHeader>
          <h2 className="font-medium">Invite a teammate</h2>
          <p className="text-xs text-slate-400">
            We&apos;ll email them an invite. If it can&apos;t be sent, you&apos;ll get a link to share yourself.
          </p>
        </CardHeader>
        <CardBody className="space-y-4">
          <form onSubmit={submitInvite} className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1">
              <Label>Email</Label>
              <Input
                data-testid="invite-email"
                type="email"
                value={inviteEmail}
                onChange={(e: any) => setInviteEmail(e.target.value)}
                placeholder="teammate@company.com"
                required
              />
            </div>
            <div className="sm:w-40">
              <Label>Role</Label>
              <select
                data-testid="invite-role"
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value as MemberRole)}
                className="w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 outline-none focus:ring-2 focus:ring-slate-600"
              >
                <option value="member">Member</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <Button data-testid="invite-submit" type="submit" disabled={inviting}>
              {inviting ? "Sending…" : "Send invite"}
            </Button>
          </form>

          {newInvite && (
            <div
              data-testid="invite-link-panel"
              className="rounded-md border border-slate-800 bg-slate-950 p-3"
            >
              {newInvite.link ? (
                <>
                  <p className="mb-2 text-xs text-slate-400">
                    Invite for <span className="text-slate-200">{newInvite.email}</span> ({ROLE_LABEL[newInvite.role]}).
                    The invite email couldn&apos;t be sent — share this link:
                  </p>
                  <div className="flex items-center gap-2">
                    <code
                      data-testid="invite-link"
                      className="min-w-0 flex-1 truncate rounded bg-slate-800 px-2 py-1.5 text-xs text-slate-100"
                    >
                      {newInvite.link}
                    </code>
                    <CopyButton value={newInvite.link} testId="invite-link-copy" />
                  </div>
                </>
              ) : (
                <p className="text-xs text-slate-400">
                  Invite emailed to <span className="text-slate-200">{newInvite.email}</span> ({ROLE_LABEL[newInvite.role]}).
                </p>
              )}
            </div>
          )}
        </CardBody>
      </Card>

      {/* Pending invites */}
      {invites.length > 0 && (
        <Card className="mb-6">
          <CardHeader>
            <h2 className="font-medium">Pending invites</h2>
          </CardHeader>
          <CardBody className="space-y-2">
            {invites.map((inv) => (
              <PendingInviteRow key={inv.id} invite={inv} onChanged={load} />
            ))}
          </CardBody>
        </Card>
      )}

      {/* Members */}
      <Card>
        <CardHeader>
          <h2 className="font-medium">Members</h2>
        </CardHeader>
        <CardBody>
          {loading ? (
            <p className="text-sm text-slate-400">Loading…</p>
          ) : members.length === 0 ? (
            <p className="text-sm text-slate-400">No members yet.</p>
          ) : (
            <ul data-testid="member-list" className="divide-y divide-slate-800">
              {members.map((m) => (
                <MemberRow key={m.id} member={m} isSelf={m.id === me?.id} onChanged={load} />
              ))}
            </ul>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

function PendingInviteRow({ invite, onChanged }: { invite: Invite; onChanged: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);

  async function revoke() {
    setBusy(true);
    try {
      await api.revokeInvite(invite.id);
      await onChanged();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      data-testid="pending-invite"
      className="flex flex-col gap-2 py-2 sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="min-w-0">
        <div className="truncate text-sm text-slate-100">{invite.email}</div>
        <div className="text-xs text-slate-400">Invited · {ROLE_LABEL[invite.role] || invite.role}</div>
      </div>
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="danger"
          data-testid="revoke-invite"
          disabled={busy}
          onClick={revoke}
        >
          {busy ? "…" : "Revoke"}
        </Button>
      </div>
    </div>
  );
}

function MemberRow({
  member,
  isSelf,
  onChanged,
}: {
  member: Member;
  isSelf: boolean;
  onChanged: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);

  async function changeRole(role: MemberRole) {
    if (role === member.role) return;
    setBusy(true);
    try {
      await api.updateMemberRole(member.id, role);
      await onChanged();
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    setBusy(true);
    try {
      await api.removeMember(member.id);
      await onChanged();
    } finally {
      setBusy(false);
      setConfirmRemove(false);
    }
  }

  return (
    <li data-testid="member-row" className="flex flex-col gap-2 py-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm text-slate-100">{member.email}</span>
          {isSelf && <Badge color="blue">You</Badge>}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* Change role. You can't demote yourself (avoids locking the tenant out of admin by accident). */}
        <select
          data-testid="member-role"
          value={member.role}
          disabled={busy || isSelf}
          onChange={(e) => changeRole(e.target.value as MemberRole)}
          className={cn(
            "rounded-md border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-100 outline-none focus:ring-2 focus:ring-slate-600",
            (busy || isSelf) && "opacity-50"
          )}
          title={isSelf ? "You can't change your own role" : undefined}
        >
          <option value="member">Member</option>
          <option value="admin">Admin</option>
        </select>

        {/* Remove member (not yourself). */}
        {!isSelf &&
          (confirmRemove ? (
            <span className="flex items-center gap-2 text-xs">
              <button
                onClick={remove}
                disabled={busy}
                data-testid="confirm-remove-member"
                className="font-medium text-red-400 hover:underline"
              >
                {busy ? "…" : "Remove"}
              </button>
              <button
                onClick={() => setConfirmRemove(false)}
                className="text-slate-400 hover:underline"
              >
                Cancel
              </button>
            </span>
          ) : (
            <button
              onClick={() => setConfirmRemove(true)}
              data-testid="remove-member"
              className="text-xs text-slate-400 hover:text-red-400"
            >
              Remove
            </button>
          ))}
      </div>
    </li>
  );
}
