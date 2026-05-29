export default function TimelineLoading() {
  return (
    <div className="space-y-4">
      <div className="skeleton h-6 w-52" />
      <div className="space-y-4">
        <div className="skeleton h-32 rounded-xl" />
        <div className="skeleton h-32 rounded-xl" />
        <div className="skeleton h-32 rounded-xl" />
      </div>
    </div>
  );
}
