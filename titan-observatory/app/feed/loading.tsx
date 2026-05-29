export default function FeedLoading() {
  return (
    <div className="space-y-4">
      <div className="skeleton h-6 w-32" />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[600px]">
        <div className="skeleton rounded-xl" />
        <div className="skeleton rounded-xl" />
        <div className="skeleton rounded-xl" />
      </div>
    </div>
  );
}
