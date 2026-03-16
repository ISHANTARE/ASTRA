/**
 * Bottom timeline slider.
 * Controls simulation time from T=0 to T+24h in 5-minute increments.
 * Per doc 05: Bottom Timeline Slider controls the simulation time.
 */
export default function TimelineSlider({ timeStep, setTimeStep, maxSteps = 288 }) {
  // Convert step index to HH:MM display
  const formatTime = (step) => {
    const totalMinutes = step * 5;
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return `T+${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
  };

  return (
    <div className="panel timeline-bar" id="timeline-bar">
      <span className="timeline-label">T+00:00</span>
      <input
        type="range"
        id="timeline-slider"
        min={0}
        max={maxSteps - 1}
        step={1}
        value={timeStep}
        onChange={(e) => setTimeStep(Number(e.target.value))}
      />
      <span className="timeline-time">{formatTime(timeStep)}</span>
    </div>
  );
}
