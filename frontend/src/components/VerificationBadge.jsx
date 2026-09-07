// Displays the citation verification status with an icon and color.
const STATUS_META = {
  SUPPORTED: {
    label: "Citation Verified",
    icon: "✓",
    classes: "bg-emerald-50 text-emerald-700 border-emerald-200",
  },
  PARTIALLY_SUPPORTED: {
    label: "Citation Partially Supported",
    icon: "⚠",
    classes: "bg-amber-50 text-amber-700 border-amber-200",
  },
  NOT_SUPPORTED: {
    label: "Citation Not Supported",
    icon: "✕",
    classes: "bg-rose-50 text-rose-700 border-rose-200",
  },
};

export default function VerificationBadge({ status, summary }) {
  // Verification only applies to a generated, cited answer. For refusals or
  // LLM-unavailable states there is no claim to verify, so render nothing.
  if (!status || status === "NOT_APPLICABLE") return null;
  const meta = STATUS_META[status] || STATUS_META.NOT_SUPPORTED;
  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium ${meta.classes}`}
      title={summary || ""}
    >
      <span aria-hidden>{meta.icon}</span>
      <span>{meta.label}</span>
    </div>
  );
}
