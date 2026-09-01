"use client";
// Documents section: read-only list of the corpus's files. Adding/removing docs
// (and re-training) is done back in the wizard, linked here.
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api } from "@/lib/api";
import { fmtBytes, PARSE_BADGE } from "@/lib/format";
import { Badge, Button, Card, CardBody, CardHeader } from "@/components/ui";
import type { Document } from "@/lib/types";

export default function DocumentsPage() {
  const { id } = useParams() as { id: string };
  const [docs, setDocs] = useState<Document[]>([]);

  useEffect(() => {
    api.listDocuments(id).then(setDocs).catch(() => setDocs([]));
  }, [id]);

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <h2 className="font-medium">Documents</h2>
        <span className="text-xs text-slate-400">{docs.length} file(s)</span>
      </CardHeader>
      <CardBody className="space-y-4">
        <ul data-testid="doc-list" className="max-h-96 divide-y divide-slate-800 overflow-y-auto text-sm">
          {docs.map((d) => {
            const badge = d.parse_status ? PARSE_BADGE[d.parse_status] : null;
            return (
              <li key={d.id} className="py-1.5" data-testid={`doc-row-${d.id}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate" title={d.filename}>{d.filename}</span>
                  <div className="flex shrink-0 items-center gap-2">
                    {badge && (
                      <span data-testid={`doc-status-${d.id}`}>
                        <Badge color={badge.color}>{badge.label}</Badge>
                      </span>
                    )}
                    <span className="text-slate-500">{fmtBytes(d.size)}</span>
                  </div>
                </div>
                {d.parse_status === "failed" && d.parse_error && (
                  <p data-testid={`doc-error-${d.id}`} className="mt-0.5 truncate text-xs text-red-400" title={d.parse_error}>
                    {d.parse_error}
                  </p>
                )}
              </li>
            );
          })}
          {docs.length === 0 && <li className="py-1.5 text-slate-500">No documents.</li>}
        </ul>
        <Link href={`/corpus/${id}/setup`}>
          <Button variant="outline" data-testid="add-docs">Add documents / re-train</Button>
        </Link>
      </CardBody>
    </Card>
  );
}
