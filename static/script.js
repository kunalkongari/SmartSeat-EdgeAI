/**
 * script.js
 * Polls /api/status every second and updates the dashboard.
 * Handles MJPEG stream reconnection automatically.
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $totalSeats    = document.getElementById("totalSeats");
const $occupiedSeats = document.getElementById("occupiedSeats");
const $vacantSeats   = document.getElementById("vacantSeats");
const $fpsValue      = document.getElementById("fpsValue");
const $fpsBadge      = document.getElementById("fpsBadge");
const $occPercent    = document.getElementById("occPercent");
const $occBarFill    = document.getElementById("occBarFill");
const $seatGrid      = document.getElementById("seatGrid");
const $statusDot     = document.getElementById("statusDot");
const $statusText    = document.getElementById("statusText");
const $errorBanner   = document.getElementById("errorBanner");
const $errorText     = document.getElementById("errorText");
const $videoOverlay  = document.getElementById("videoOverlay");
const $lastUpdate    = document.getElementById("lastUpdate");
const $videoFeed     = document.getElementById("videoFeed");

// ── State ─────────────────────────────────────────────────────────────────────
let prevSeatCount = -1;
let pollFailures  = 0;
let videoReady    = false;

// ── Helpers ───────────────────────────────────────────────────────────────────
function seatIcon(occupied) {
  return occupied ? "🪑" : "⬜";
}

function pad(n) {
  return String(n).padStart(2, "0");
}

function timeNow() {
  const d = new Date();
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

// ── Seat grid ─────────────────────────────────────────────────────────────────
function updateSeatGrid(seats) {
  if (!seats || seats.length === 0) {
    if (prevSeatCount !== 0) {
      $seatGrid.innerHTML = '<div class="no-seats">No seats detected yet…</div>';
      prevSeatCount = 0;
    }
    return;
  }

  // Full rebuild only when seat count changes (avoids flicker)
  if (seats.length !== prevSeatCount) {
    $seatGrid.innerHTML = "";
    seats.forEach(seat => {
      const box       = document.createElement("div");
      box.id          = `seat-${seat.seat_id}`;
      box.className   = `seat-box ${seat.occupied ? "occupied" : "vacant"}`;
      box.innerHTML   = `
        <span class="seat-icon">${seatIcon(seat.occupied)}</span>
        <div class="seat-number">Seat ${seat.seat_id}</div>
        <div class="seat-status">${seat.occupied ? "OCCUPIED" : "VACANT"}</div>
      `;
      $seatGrid.appendChild(box);
    });
    prevSeatCount = seats.length;
    return;
  }

  // Incremental update — only changed boxes
  seats.forEach(seat => {
    const box = document.getElementById(`seat-${seat.seat_id}`);
    if (!box) return;
    const wasOcc = box.classList.contains("occupied");
    if (wasOcc !== seat.occupied) {
      box.className = `seat-box ${seat.occupied ? "occupied" : "vacant"}`;
      box.querySelector(".seat-icon").textContent   = seatIcon(seat.occupied);
      box.querySelector(".seat-status").textContent = seat.occupied ? "OCCUPIED" : "VACANT";
    }
  });
}

// ── Smooth counter ────────────────────────────────────────────────────────────
function animateNumber(el, target) {
  const current = parseInt(el.textContent, 10);
  if (isNaN(current) || current === target) { el.textContent = target; return; }
  const step  = target > current ? 1 : -1;
  let   val   = current;
  const timer = setInterval(() => {
    val += step;
    el.textContent = val;
    if (val === target) clearInterval(timer);
  }, 35);
}

// ── Apply status data ─────────────────────────────────────────────────────────
function applyStatus(data) {
  animateNumber($totalSeats,    data.total_seats);
  animateNumber($occupiedSeats, data.occupied_seats);
  animateNumber($vacantSeats,   data.vacant_seats);

  const fps = typeof data.fps === "number" ? data.fps.toFixed(1) : "—";
  $fpsValue.textContent = fps;
  $fpsBadge.textContent = `${fps} FPS`;

  const pct = data.total_seats > 0
    ? Math.round((data.occupied_seats / data.total_seats) * 100)
    : 0;
  $occPercent.textContent  = `${pct} %`;
  $occBarFill.style.width  = `${pct}%`;

  updateSeatGrid(data.seats || []);

  // Connection status
  $statusDot.className    = "status-dot live";
  $statusText.textContent = "Live";
  $lastUpdate.textContent = timeNow();
  pollFailures = 0;

  // Error banner
  if (data.error) {
    $errorText.textContent = data.error;
    $errorBanner.classList.remove("hidden");
  } else {
    $errorBanner.classList.add("hidden");
  }
}

// ── Poll /api/status ──────────────────────────────────────────────────────────
async function poll() {
  try {
    const res  = await fetch("/api/status", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    applyStatus(data);
  } catch (err) {
    pollFailures++;
    $statusDot.className    = "status-dot error";
    $statusText.textContent = "Disconnected";
    if (pollFailures === 1) console.warn("Poll error:", err);
  }
}

// ── MJPEG stream handling ─────────────────────────────────────────────────────
$videoFeed.addEventListener("load", () => {
  if (!videoReady) {
    $videoOverlay.classList.add("hidden");
    videoReady = true;
  }
});

$videoFeed.addEventListener("error", () => {
  $videoOverlay.classList.remove("hidden");
  $videoOverlay.querySelector("span").textContent = "Reconnecting…";
  // Retry stream after 2 s
  setTimeout(() => {
    $videoFeed.src = `/video?t=${Date.now()}`;
  }, 2000);
});

// Hide overlay once first frame arrives (MJPEG keeps "loading" permanently)
// Use a polling trick to detect the first painted frame
let checkInterval = setInterval(() => {
  if ($videoFeed.naturalWidth > 0) {
    $videoOverlay.classList.add("hidden");
    videoReady = true;
    clearInterval(checkInterval);
  }
}, 500);

// ── Kick off ──────────────────────────────────────────────────────────────────
poll();
setInterval(poll, 1000);
