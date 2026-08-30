"use client";
// Documents section: read-only list of the corpus's files. Adding/removing docs
// (and re-training) is done back in the wizard, linked here.
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api } from "@/lib/api";
import { Button, Card, CardBody, CardHeader } from "@/components/ui";

export default function DocumentsPage() {
  const { id } = useParams() as { id: string };
  const [docs, setDocs] = useState<any[]>([]);

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
          {docs.map((d) => (
            <li key={d.id} className="flex justify-between gap-2 py-1.5">
              <span className="truncate" title={d.filename}>{d.filename}</span>
              <span className="shrink-0 text-slate-500">{d.size} B</span>
            </li>
          ))}
          {docs.length === 0 && <li className="py-1.5 text-slate-500">No documents.</li>}
        </ul>
        <Link href={`/corpus/${id}/setup`}>
          <Button variant="outline" data-testid="add-docs">Add documents / re-train</Button>
        </Link>
      </CardBody>
    </Card>
  );
}
