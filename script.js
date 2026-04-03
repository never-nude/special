import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ─────────────────────────────────────────────
// SHARED NOISE GLSL
// ─────────────────────────────────────────────

const NOISE_GLSL = /* glsl */ `
vec3 mod289(vec3 x){ return x - floor(x*(1.0/289.0))*289.0; }
vec2 mod289(vec2 x){ return x - floor(x*(1.0/289.0))*289.0; }
vec3 permute(vec3 x){ return mod289(((x*34.0)+10.0)*x); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                      -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                           + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                           dot(x12.zw,x12.zw)), 0.0);
  m = m*m; m = m*m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float fbm(vec2 p){
  float f = 0.0, w = 0.5;
  for(int i = 0; i < 5; i++){ f += w * snoise(p); p *= 2.0; w *= 0.5; }
  return f;
}

float fbm3(vec2 p){
  float f = 0.0, w = 0.5;
  for(int i = 0; i < 3; i++){ f += w * snoise(p); p *= 2.0; w *= 0.5; }
  return f;
}
`;

// ─────────────────────────────────────────────
// RENDERER
// ─────────────────────────────────────────────

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// ─────────────────────────────────────────────
// SCENE & CAMERA
// ─────────────────────────────────────────────

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0xbfbbb5, 0.014);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 5, 0);

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────

const WORLD_H = 200;       // altitude 1.0 = y 200
const CLOUD_TOP = 68;      // clouds from y=0 to y=68 (~alt 0.34)
const CLOUD_COUNT = 45;
const CLOUD_SPREAD_Z = 500;
const CLOUD_SPREAD_X = 300;
const STAR_COUNT = 350;
const BASE_DRIFT = 0.12;   // forward speed (units/sec)

// ─────────────────────────────────────────────
// SKY DOME
// ─────────────────────────────────────────────

const skyGeo = new THREE.SphereGeometry(900, 32, 32);
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide,
  depthWrite: false,
  fog: false,
  uniforms: {
    uAlt: { value: 0 },
    uTime: { value: 0 },
  },
  vertexShader: /* glsl */ `
    varying vec3 vDir;
    void main(){
      vDir = normalize(position);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform float uAlt;
    uniform float uTime;
    varying vec3 vDir;

    void main(){
      float y = vDir.y * 0.5 + 0.5;
      float alt = uAlt;

      // palette per altitude band
      vec3 fogCol  = vec3(0.75, 0.73, 0.70);
      vec3 lowBot  = vec3(0.82, 0.78, 0.72);
      vec3 lowTop  = vec3(0.58, 0.68, 0.82);
      vec3 midBot  = vec3(0.90, 0.70, 0.42);
      vec3 midTop  = vec3(0.25, 0.45, 0.78);
      vec3 highBot = vec3(0.15, 0.25, 0.50);
      vec3 highTop = vec3(0.05, 0.08, 0.18);
      vec3 spaceA  = vec3(0.04, 0.06, 0.12);
      vec3 spaceB  = vec3(0.02, 0.02, 0.05);

      float bFog   = 1.0 - smoothstep(0.0,  0.2,  alt);
      float bLow   = smoothstep(0.0, 0.15, alt) * (1.0 - smoothstep(0.25, 0.5,  alt));
      float bMid   = smoothstep(0.25, 0.45, alt) * (1.0 - smoothstep(0.6,  0.8,  alt));
      float bHigh  = smoothstep(0.6, 0.75, alt) * (1.0 - smoothstep(0.85, 0.98, alt));
      float bSpace = smoothstep(0.85, 0.98, alt);

      vec3 sky = vec3(0.0);
      sky += bFog   * fogCol;
      sky += bLow   * mix(lowBot,  lowTop,  y);
      sky += bMid   * mix(midBot,  midTop,  y);
      sky += bHigh  * mix(highBot, highTop, y);
      sky += bSpace * mix(spaceA,  spaceB,  y);
      sky /= max(bFog + bLow + bMid + bHigh + bSpace, 0.001);

      // horizon glow band
      float hVis  = smoothstep(0.2, 0.45, alt) * (1.0 - smoothstep(0.9, 1.0, alt));
      float hBand = exp(-pow(y - 0.5, 2.0) * 50.0);
      vec3  hCol  = mix(
        mix(vec3(0.95, 0.88, 0.78), vec3(1.0, 0.72, 0.42), smoothstep(0.2, 0.5, alt)),
        vec3(0.4, 0.6, 1.0),
        smoothstep(0.6, 0.9, alt)
      );
      sky += hCol * hBand * hVis * 0.3;

      // atmospheric limb at orbit
      float limbVis  = smoothstep(0.75, 0.95, alt);
      float limbBand = exp(-pow(y - 0.43, 2.0) * 250.0);
      sky += vec3(0.3, 0.5, 0.9) * limbBand * limbVis * 0.4;

      // earth surface implied below at orbit
      float earthVis = smoothstep(0.78, 0.98, alt);
      float below    = smoothstep(0.42, 0.25, y);
      vec3  earthCol = mix(vec3(0.08, 0.15, 0.30), vec3(0.03, 0.05, 0.10), below);
      sky = mix(sky, earthCol, below * earthVis * 0.7);

      // breathing
      sky *= 1.0 + sin(uTime * 0.25) * 0.01;

      gl_FragColor = vec4(sky, 1.0);
    }
  `,
});

const skyDome = new THREE.Mesh(skyGeo, skyMat);
scene.add(skyDome);

// ─────────────────────────────────────────────
// CLOUD SYSTEM
// ─────────────────────────────────────────────

function makeCloudMaterial(seed, scale) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: true,
    uniforms: {
      ...THREE.UniformsLib.fog,
      uTime: { value: 0 },
      uOpacity: { value: 1 },
      uColor: { value: new THREE.Color(0.92, 0.90, 0.87) },
      uScale: { value: scale },
      uSeed: { value: seed },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying float vFogDepth;
      void main(){
        vUv = uv;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vFogDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float fogDensity;
      uniform vec3 fogColor;
      ${NOISE_GLSL}
      uniform float uTime;
      uniform float uOpacity;
      uniform vec3 uColor;
      uniform float uScale;
      uniform float uSeed;
      varying vec2 vUv;
      varying float vFogDepth;

      void main(){
        vec2 p = (vUv - 0.5) * uScale + uSeed;
        p += vec2(uTime * 0.006, uTime * 0.003);

        float n = fbm(p) * 0.5 + 0.5;
        n = smoothstep(0.32, 0.72, n);

        // soft circular falloff
        float d = length(vUv - 0.5) * 2.0;
        float edge = 1.0 - smoothstep(0.55, 1.0, d);

        // slight internal color variation
        float detail = snoise(p * 2.0) * 0.04;
        vec3 col = uColor + detail;

        float alpha = n * edge * uOpacity;

        // manual fog integration
        float fogFactor = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
        col = mix(col, fogColor, fogFactor);
        alpha *= 1.0 - fogFactor * 0.6;

        gl_FragColor = vec4(col, alpha);
      }
    `,
  });
}

const clouds = [];

for (let i = 0; i < CLOUD_COUNT; i++) {
  const seed = Math.random() * 100;
  const scale = 1.5 + Math.random() * 2.5;
  const mat = makeCloudMaterial(seed, scale);

  // vary color: darker at bottom, lighter up top
  const heightFrac = Math.random();
  const yPos = Math.pow(heightFrac, 0.6) * CLOUD_TOP;
  const brightness = 0.82 + heightFrac * 0.13;
  mat.uniforms.uColor.value.setRGB(brightness, brightness - 0.02, brightness - 0.05);

  const size = 70 + Math.random() * 80;
  const geo = new THREE.PlaneGeometry(size, size);
  const mesh = new THREE.Mesh(geo, mat);

  mesh.position.set(
    (Math.random() - 0.5) * CLOUD_SPREAD_X,
    yPos,
    Math.random() * CLOUD_SPREAD_Z
  );

  // mix of horizontal and tilted orientations
  if (i < CLOUD_COUNT * 0.6) {
    // mostly horizontal (cloud layer seen from above/below)
    mesh.rotation.x = -Math.PI / 2 + (Math.random() - 0.5) * 0.35;
  } else {
    // tilted toward vertical (visible when looking forward in fog)
    mesh.rotation.x = -Math.PI / 4 + (Math.random() - 0.5) * 0.6;
    mesh.rotation.y = Math.random() * Math.PI * 2;
  }
  mesh.rotation.z = Math.random() * Math.PI * 2;

  scene.add(mesh);
  clouds.push(mesh);
}

// ─────────────────────────────────────────────
// STARS
// ─────────────────────────────────────────────

const starPositions = new Float32Array(STAR_COUNT * 3);
const starSizes = new Float32Array(STAR_COUNT);
const starPhases = new Float32Array(STAR_COUNT);

for (let i = 0; i < STAR_COUNT; i++) {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(1 - Math.random() * 1.2); // bias upper hemisphere
  const r = 650;
  starPositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
  starPositions[i * 3 + 1] = Math.abs(r * Math.cos(phi));
  starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  starSizes[i] = 1.0 + Math.random() * 3.0;
  starPhases[i] = Math.random() * Math.PI * 2;
}

const starGeo = new THREE.BufferGeometry();
starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
starGeo.setAttribute('aSize', new THREE.BufferAttribute(starSizes, 1));
starGeo.setAttribute('aPhase', new THREE.BufferAttribute(starPhases, 1));

const starMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  fog: false,
  uniforms: {
    uOpacity: { value: 0 },
    uTime: { value: 0 },
  },
  vertexShader: /* glsl */ `
    attribute float aSize;
    attribute float aPhase;
    varying float vTwinkle;
    uniform float uTime;
    void main(){
      vTwinkle = 0.65 + 0.35 * sin(uTime * (0.4 + aSize * 0.2) + aPhase);
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = aSize * (350.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: /* glsl */ `
    uniform float uOpacity;
    varying float vTwinkle;
    void main(){
      float d = length(gl_PointCoord - 0.5) * 2.0;
      float glow = exp(-d * d * 4.0);
      gl_FragColor = vec4(vec3(0.9, 0.92, 1.0), glow * uOpacity * vTwinkle);
    }
  `,
});

const stars = new THREE.Points(starGeo, starMat);
scene.add(stars);

// ─────────────────────────────────────────────
// POST-PROCESSING
// ─────────────────────────────────────────────

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloom = new UnrealBloomPass(
  new THREE.Vector2(innerWidth, innerHeight),
  0.55,  // strength
  1.3,   // radius (wide = soft)
  0.25   // threshold (low = more glow)
);
composer.addPass(bloom);

// vignette + ghosting + warmth
const postShader = {
  uniforms: {
    tDiffuse: { value: null },
    uGhost: { value: 0.035 },
    uGhostOffset: { value: new THREE.Vector2(0.003, 0.0015) },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uGhost;
    uniform vec2 uGhostOffset;
    varying vec2 vUv;
    void main(){
      vec4 color = texture2D(tDiffuse, vUv);

      // subtle ghosting — makes double vision feel intentional
      vec4 ghost = texture2D(tDiffuse, vUv + uGhostOffset);
      color = mix(color, ghost, uGhost);

      // vignette (wide, soft)
      vec2 vc = vUv - 0.5;
      float vig = 1.0 - dot(vc, vc) * 0.55;
      color.rgb *= smoothstep(0.0, 1.0, vig);

      // slight warmth at edges
      float r2 = dot(vc, vc);
      color.r *= 1.0 + r2 * 0.06;
      color.b *= 1.0 - r2 * 0.03;

      gl_FragColor = color;
    }
  `,
};
composer.addPass(new ShaderPass(postShader));
composer.addPass(new OutputPass());

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

const S = {
  altitude: 0,
  targetAlt: 0,
  time: 0,
  lastTime: performance.now(),

  // mouse influence
  mouseX: 0,
  mouseY: 0,
  smoothX: 0,
  smoothY: 0,

  // camera look drift
  lookX: 0,
  lookY: 0,

  // continuous forward travel
  travelZ: 0,

  // drag state
  dragging: false,
  dragX: 0,
  dragY: 0,
};

// ─────────────────────────────────────────────
// INPUT
// ─────────────────────────────────────────────

addEventListener('mousemove', e => {
  S.mouseX = (e.clientX / innerWidth) * 2 - 1;
  S.mouseY = -((e.clientY / innerHeight) * 2 - 1);
});

addEventListener('mousedown', () => { S.dragging = true; });
addEventListener('mouseup', () => { S.dragging = false; });

addEventListener('wheel', e => {
  e.preventDefault();
  S.targetAlt += e.deltaY * 0.0012;
  S.targetAlt = Math.max(0, Math.min(1, S.targetAlt));
}, { passive: false });

// touch
let touchY0 = 0;
addEventListener('touchstart', e => {
  touchY0 = e.touches[0].clientY;
  S.dragging = true;
}, { passive: true });

addEventListener('touchend', () => { S.dragging = false; }, { passive: true });

addEventListener('touchmove', e => {
  if (e.touches.length >= 1) {
    S.mouseX = (e.touches[0].clientX / innerWidth) * 2 - 1;
    S.mouseY = -((e.touches[0].clientY / innerHeight) * 2 - 1);
    const dy = touchY0 - e.touches[0].clientY;
    touchY0 = e.touches[0].clientY;
    S.targetAlt += dy * 0.005;
    S.targetAlt = Math.max(0, Math.min(1, S.targetAlt));
  }
}, { passive: true });

// ─────────────────────────────────────────────
// RESIZE
// ─────────────────────────────────────────────

function onResize() {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
}
addEventListener('resize', onResize);

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────

function expSmooth(current, target, rate, dt) {
  return current + (target - current) * (1 - Math.exp(-rate * dt));
}

// ─────────────────────────────────────────────
// ANIMATION LOOP
// ─────────────────────────────────────────────

function frame(now) {
  requestAnimationFrame(frame);

  const dt = Math.min((now - S.lastTime) / 1000, 0.1);
  S.lastTime = now;
  S.time += dt;

  const alt = S.altitude = expSmooth(S.altitude, S.targetAlt, 3.5, dt);

  // ── mouse smoothing (very heavy) ──
  const mouseRate = S.dragging ? 1.5 : 0.8;
  S.smoothX = expSmooth(S.smoothX, S.mouseX, mouseRate, dt);
  S.smoothY = expSmooth(S.smoothY, S.mouseY, mouseRate, dt);

  // ── autonomous sway ──
  const swayX = Math.sin(S.time * 0.1) * 0.04
              + Math.sin(S.time * 0.073) * 0.025;
  const swayY = Math.cos(S.time * 0.083) * 0.02
              + Math.sin(S.time * 0.131) * 0.015;

  // ── camera position ──
  // forward drift (always, feels like current carrying you)
  S.travelZ += BASE_DRIFT * (1 + alt * 0.4) * dt;

  // altitude → Y
  const targetY = alt * WORLD_H;
  camera.position.y = expSmooth(camera.position.y, targetY, 2.5, dt);
  camera.position.z = S.travelZ;

  // subtle lateral drift from mouse + sway
  const lateralInfluence = S.dragging ? 0.6 : 0.25;
  camera.position.x += (S.smoothX * lateralInfluence + swayX) * dt;
  // gentle re-centering
  camera.position.x *= Math.pow(0.992, dt * 60);

  // ── camera look ──
  const lookInfluence = S.dragging ? 0.45 : 0.25;
  S.lookX = expSmooth(S.lookX, S.smoothX * lookInfluence + swayX, 0.6, dt);
  S.lookY = expSmooth(S.lookY, S.smoothY * 0.15 + swayY, 0.6, dt);

  camera.lookAt(
    camera.position.x + S.lookX * 3,
    camera.position.y + S.lookY * 1.5,
    camera.position.z + 12
  );

  // ── sky dome follows camera ──
  skyDome.position.copy(camera.position);
  skyMat.uniforms.uAlt.value = alt;
  skyMat.uniforms.uTime.value = S.time;

  // ── stars follow camera ──
  stars.position.copy(camera.position);
  const starOpacity = Math.pow(Math.max(0, (alt - 0.5) / 0.5), 2.2);
  starMat.uniforms.uOpacity.value = starOpacity;
  starMat.uniforms.uTime.value = S.time;

  // ── fog ──
  const fogDensity = THREE.MathUtils.lerp(0.016, 0.0004, Math.pow(alt, 0.65));
  scene.fog.density = fogDensity;

  // fog color blends with altitude
  const fogLow  = new THREE.Color(0.75, 0.73, 0.70);
  const fogMid  = new THREE.Color(0.52, 0.58, 0.68);
  const fogHigh = new THREE.Color(0.06, 0.08, 0.15);
  if (alt < 0.5) {
    scene.fog.color.copy(fogLow).lerp(fogMid, alt * 2);
  } else {
    scene.fog.color.copy(fogMid).lerp(fogHigh, (alt - 0.5) * 2);
  }

  // ── clouds ──
  const cloudOpacity = Math.max(0, 1 - Math.pow(Math.max(0, alt - 0.05) / 0.55, 1.6));

  for (const cloud of clouds) {
    const mat = cloud.material;
    mat.uniforms.uTime.value = S.time;
    mat.uniforms.uOpacity.value = cloudOpacity;
    mat.uniforms.fogDensity.value = scene.fog.density;
    mat.uniforms.fogColor.value.copy(scene.fog.color);

    // recycle clouds that drift behind camera
    if (cloud.position.z < camera.position.z - 100) {
      cloud.position.z += CLOUD_SPREAD_Z + Math.random() * 60;
      cloud.position.x = (Math.random() - 0.5) * CLOUD_SPREAD_X;
    }
  }

  // ── bloom adjusts with altitude ──
  bloom.strength = 0.45 + alt * 0.25;

  // ── render ──
  composer.render();
}

S.lastTime = performance.now();
requestAnimationFrame(frame);
