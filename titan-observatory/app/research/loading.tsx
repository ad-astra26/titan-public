export default function ResearchLoading() {
  return (
    <div className="space-y-6">
      <div className="skeleton h-6 w-40" />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="skeleton h-64 rounded-xl" />
        <div className="skeleton h-64 rounded-xl" />
      </div>
      <div className="skeleton h-48 rounded-xl" />
    </div>
  );
}
