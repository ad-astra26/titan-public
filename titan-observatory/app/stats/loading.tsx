export default function StatsLoading() {
  return (
    <div className="space-y-6">
      <div className="skeleton h-6 w-36" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="skeleton h-24 rounded-xl" />
        <div className="skeleton h-24 rounded-xl" />
        <div className="skeleton h-24 rounded-xl" />
        <div className="skeleton h-24 rounded-xl" />
      </div>
      <div className="skeleton h-48 rounded-xl" />
    </div>
  );
}
