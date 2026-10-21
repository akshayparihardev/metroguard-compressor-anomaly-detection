// Scroll motion with GSAP and ScrollTrigger (vendored, loaded on demand). Everything here is decoration: the
// page is complete without it, and nothing runs when the visitor prefers reduced motion.

const scripts = new Map();

function loadScript(src) {
  if (!scripts.has(src)) {
    scripts.set(src, new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = new URL(src, import.meta.url).href;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Could not load ${src}`));
      document.head.append(script);
    }));
  }
  return scripts.get(src);
}

export const motionAllowed = () => !matchMedia("(prefers-reduced-motion: reduce)").matches;

async function loadGsap() {
  await loadScript("./vendor/gsap.min.js");
  await loadScript("./vendor/ScrollTrigger.min.js");
  window.gsap.registerPlugin(window.ScrollTrigger);
  return { gsap: window.gsap, ScrollTrigger: window.ScrollTrigger };
}

/** Count a number up from zero, keeping the element's final text (and its formatting) exact. */
function countUp(gsap, node) {
  const text = node.textContent;
  const match = text.match(/[\d,]+(\.\d+)?/);
  if (!match) return;
  const target = Number(match[0].replace(/,/g, ""));
  const decimals = match[1] ? match[1].length - 1 : 0;
  const state = { value: 0 };
  gsap.to(state, {
    value: target, duration: 1.1, ease: "power2.out",
    onUpdate: () => {
      const shown = state.value.toLocaleString("en-GB", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
      node.textContent = text.replace(match[0], shown);
    },
    onComplete: () => { node.textContent = text; },
  });
}

/**
 * Overview journey: steps rise in as they enter, marked numbers count up, verdict bars grow, and the step bar
 * highlights the step being read.
 */
export async function animateJourney(root) {
  if (!motionAllowed()) return;
  const { gsap, ScrollTrigger } = await loadGsap();
  // Content is never hidden up front: each step animates *from* faded only as it scrolls in, so the page stays
  // complete if animation frames never run (printing, a background tab, a blocked script).
  const steps = [...root.querySelectorAll(".journey-step")];
  const below = steps.filter((stepNode) => stepNode.getBoundingClientRect().top > innerHeight);
  ScrollTrigger.batch(below, {
    start: "top 92%",
    once: true,
    onEnter: (batch) => {
      gsap.from(batch, { autoAlpha: 0, y: 28, duration: 0.6, ease: "power3.out", stagger: 0.08 });
      batch.forEach((stepNode) => stepNode.querySelectorAll("[data-count]").forEach((n) => countUp(gsap, n)));
    },
  });

  const bars = root.querySelectorAll(".verdict-bar .meter i");
  ScrollTrigger.create({
    trigger: root.querySelector(".verdict"),
    start: "top 85%",
    once: true,
    onEnter: () => gsap.fromTo(bars, { scaleX: 0 }, { scaleX: 1, transformOrigin: "left center", duration: 0.9, ease: "power3.out", stagger: 0.12 }),
  });

  const links = new Map([...root.querySelectorAll(".flow a")].map((a) => [a.getAttribute("href").slice(1), a]));
  steps.forEach((stepNode) => {
    ScrollTrigger.create({
      trigger: stepNode,
      start: "top 45%",
      end: "bottom 45%",
      onToggle: ({ isActive }) => links.get(stepNode.id)?.classList.toggle("active", isActive),
    });
  });
}

/** Play a Lottie animation (lottie-web light, SVG renderer) in `host`; with reduced motion it shows one still frame. */
export async function playLottie(host, path) {
  await loadScript("./vendor/lottie_light.min.js");
  const animation = window.lottie.loadAnimation({
    container: host, renderer: "svg", loop: true, autoplay: motionAllowed(),
    path: new URL(path, import.meta.url).href,
  });
  if (!motionAllowed()) animation.addEventListener("DOMLoaded", () => animation.goToAndStop(30, true));
  return animation;
}
