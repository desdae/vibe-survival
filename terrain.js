import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';

// ===== CONFIGURATION =====
const CONFIG = {
    terrain: {
        size: 8000,           // 16x original 500
        segments: 800,       // Higher resolution for detail without killing FPS
        maxHeight: 600,      // Massive mountains for scale
        waterLevel: 80,      // Adjusted water level
        scale: 0.0008,       // Much smaller scale = larger features (continents)
    },
    camera: {
        fov: 75,
        near: 0.1,             // Lower near clip for viewmodels
        far: 15000,          // See far
        moveSpeed: 50.0,     // Much faster movement
        sprintMultiplier: 5, // Faster sprint
        mouseSensitivity: 0.002,
    }
};

// ===== NOISE GENERATION (Simplex-like) =====
class SimplexNoise {
    constructor(seed = Math.random()) {
        this.seed = seed;
        this.perm = this.buildPermutationTable();
    }

    buildPermutationTable() {
        const p = [];
        for (let i = 0; i < 256; i++) p[i] = i;

        // Shuffle
        for (let i = 255; i > 0; i--) {
            const n = Math.floor((this.seed * 138.7 + i * 0.314) % (i + 1));
            [p[i], p[n]] = [p[n], p[i]];
        }

        return [...p, ...p];
    }

    fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    lerp(t, a, b) {
        return a + t * (b - a);
    }

    grad(hash, x, y) {
        const h = hash & 3;
        const u = h < 2 ? x : y;
        const v = h < 2 ? y : x;
        return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
    }

    noise(x, y) {
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;

        x -= Math.floor(x);
        y -= Math.floor(y);

        const u = this.fade(x);
        const v = this.fade(y);

        const a = this.perm[X] + Y;
        const aa = this.perm[a];
        const ab = this.perm[a + 1];
        const b = this.perm[X + 1] + Y;
        const ba = this.perm[b];
        const bb = this.perm[b + 1];

        return this.lerp(v,
            this.lerp(u, this.grad(this.perm[aa], x, y),
                this.grad(this.perm[ba], x - 1, y)),
            this.lerp(u, this.grad(this.perm[ab], x, y - 1),
                this.grad(this.perm[bb], x - 1, y - 1))
        );
    }

    octaveNoise(x, y, octaves = 4, persistence = 0.5) {
        let total = 0;
        let frequency = 1;
        let amplitude = 1;
        let maxValue = 0;

        for (let i = 0; i < octaves; i++) {
            total += this.noise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= 2;
        }

        return total / maxValue;
    }
}

// ===== TERRAIN GENERATOR =====
class TerrainGenerator {
    constructor(config) {
        this.config = config;
        this.noise = new SimplexNoise(Math.random());
    }

    getElevation(x, z) {
        const { scale, maxHeight } = this.config.terrain;

        const nx = x * scale;
        const ny = z * scale;

        // Base terrain
        let height = this.noise.octaveNoise(nx, ny, 6, 0.5);

        // Mountain Mask
        let mountainMask = this.noise.octaveNoise(nx * 0.4 + 500, ny * 0.4 + 500, 3, 0.5);
        mountainMask = Math.max(0, mountainMask * 1.5 - 0.3);
        mountainMask = Math.pow(mountainMask, 1.5);

        // River/Valley formation
        let riverNoise = this.noise.octaveNoise(nx * 1.5, ny * 1.5, 4, 0.5);
        riverNoise = Math.abs(riverNoise * 2.0 - 1.0);
        riverNoise = 1.0 - Math.pow(riverNoise, 0.5);

        // Combine
        height = height * 0.9 - (riverNoise * 0.2);

        // Falloff
        const dist = Math.sqrt(x * x + z * z);
        const mapRadius = 32000;
        const falloff = Math.max(0.0, 1.0 - Math.pow(dist / mapRadius, 4.0));
        height *= falloff;

        // Remap
        height = height * 0.6 + 0.35;

        if (height > 0.45) {
            height += mountainMask * 1.2;
        }

        height = Math.pow(Math.max(0, height), 2.0);
        height = height * maxHeight;
        height = Math.max(-50, height);

        return height;
    }

    getNormal(x, z) {
        const epsilon = 1.0;
        const hL = this.getElevation(x - epsilon, z);
        const hR = this.getElevation(x + epsilon, z);
        const hD = this.getElevation(x, z - epsilon);
        const hU = this.getElevation(x, z + epsilon);

        const normal = new THREE.Vector3(hL - hR, 2.0 * epsilon, hD - hU);
        normal.normalize();
        return normal;
    }


    createTerrainMaterial() {
        const { waterLevel, maxHeight } = this.config.terrain;

        // Create procedural textures
        const sandTexture = this.createTexture('#f4e7d7', '#e5d4b5', 0.5);
        const grassTexture = this.createTexture('#4a7c3a', '#2d5a2d', 0.3); // Darker grass variety
        const stoneTexture = this.createTexture('#6b6b6b', '#4a4a4a', 0.6);
        const snowTexture = this.createTexture('#ffffff', '#e8e8e8', 0.2);

        // Increase texture repeat for huge terrain
        [sandTexture, grassTexture, stoneTexture, snowTexture].forEach(tex => {
            tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(200, 200); // Higher repeat for detail
        });

        return new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.merge([
                THREE.UniformsLib.fog,
                {
                    sandTexture: { value: sandTexture },
                    grassTexture: { value: grassTexture },
                    stoneTexture: { value: stoneTexture },
                    snowTexture: { value: snowTexture },
                    waterLevel: { value: waterLevel },
                    maxHeight: { value: maxHeight },
                }
            ]),
            vertexShader: `
                #include <common>
                #include <fog_pars_vertex>
                #include <logdepthbuf_pars_vertex>
                
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec2 vUv;
                
                void main() {
                    vPosition = position; // Local position (z is up before rotation)
                    vNormal = normal;
                    vUv = uv;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_Position = projectionMatrix * mvPosition;
                    
                    #include <logdepthbuf_vertex>
                    #include <fog_vertex>
                }
            `,
            fragmentShader: `
                #include <common>
                #include <fog_pars_fragment>
                #include <logdepthbuf_pars_fragment>
                
                uniform sampler2D sandTexture;
                uniform sampler2D grassTexture;
                uniform sampler2D stoneTexture;
                uniform sampler2D snowTexture;
                uniform float waterLevel;
                uniform float maxHeight;
                
                varying vec3 vPosition;
                varying vec3 vNormal;
                varying vec2 vUv;
                
                void main() {
                    #include <logdepthbuf_fragment>
                    
                    float height = vPosition.z;
                    vec4 color;
                    
                    // Sample textures with high frequency UVs for detail
                    vec4 sand = texture2D(sandTexture, vUv * 200.0);
                    vec4 grass = texture2D(grassTexture, vUv * 200.0);
                    vec4 stone = texture2D(stoneTexture, vUv * 200.0);
                    vec4 snow = texture2D(snowTexture, vUv * 200.0);
                    
                    // Blend based on height
                    // Beach transition
                    float sandBlend = smoothstep(waterLevel - 10.0, waterLevel + 10.0, height);
                    color = mix(sand, grass, sandBlend);
                    
                    // Mountain base
                    float grassBlend = smoothstep(maxHeight * 0.25, maxHeight * 0.45, height);
                    
                    // Slope based blending for grass/stone
                    float slope = 1.0 - vNormal.z; // 0 for flat, 1 for vertical
                    float slopeMask = smoothstep(0.1, 0.4, slope);
                    
                    // Mix grass and stone based on height AND slope
                    // Steeper slopes get stone sooner
                    vec4 terrainBase = mix(color, stone, max(grassBlend, slopeMask * 0.8));
                    
                    // Snow at peaks
                    float snowBlend = smoothstep(maxHeight * 0.7, maxHeight * 0.9, height);
                    // Snow sticks to flatter areas more? Or just covers everything at top
                    color = mix(terrainBase, snow, snowBlend);
                    
                    gl_FragColor = color;

                    #include <fog_fragment>
                }
            `,
            fog: true
        });
    }

    createTexture(color1, color2, roughness = 0.5) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');

        // Fill base
        ctx.fillStyle = color1;
        ctx.fillRect(0, 0, 256, 256);

        // Add noise pattern with wrapping
        for (let i = 0; i < 4000; i++) {
            const x = Math.random() * 256;
            const y = Math.random() * 256;
            const size = Math.random() * 2 + 1;

            ctx.fillStyle = color2;
            ctx.globalAlpha = Math.random() * roughness;

            // Draw with wrap-around
            const drawRect = (dx, dy) => {
                ctx.fillRect(dx, dy, size, size);
            };

            drawRect(x, y);
            if (x + size > 256) drawRect(x - 256, y);
            if (y + size > 256) drawRect(x, y - 256);
            if (x + size > 256 && y + size > 256) drawRect(x - 256, y - 256);
        }

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    createWaterBodies() {
        const { size, waterLevel, segments } = this.config.terrain;
        const waterGroup = new THREE.Group();

        // Main water plane
        const waterGeometry = new THREE.PlaneGeometry(size, size, 128, 128); // Higher segment count for waves
        const waterMaterial = this.createWaterMaterial();

        const water = new THREE.Mesh(waterGeometry, waterMaterial);
        water.rotation.x = -Math.PI / 2;
        water.position.y = waterLevel; // Explicitly at water level

        waterGroup.add(water);

        return waterGroup;
    }

    createWaterMaterial() {
        return new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.merge([
                THREE.UniformsLib.fog,
                {
                    time: { value: 0 },
                    waterColor: { value: new THREE.Color(0x006994) }, // Deeper blue
                    foamColor: { value: new THREE.Color(0xffffff) },
                    alpha: { value: 0.75 },
                }
            ]),
            vertexShader: `
                #include <common>
                #include <fog_pars_vertex>
                #include <logdepthbuf_pars_vertex>
                
                uniform float time;
                varying vec2 vUv;
                varying float vWave;
                
                // Simplex noise for vertex displacement could be expensive, 
                // using superposition of sine waves for efficiency and good look
                void main() {
                    vUv = uv;
                    
                    // Transform to world space for seamless waves across chunks
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vec3 pos = position; // Original local pos for gl_Position calculation
                    
                    // Use world coordinates for wave function
                    // World X and Z (horizontal plane).
                    float wx = worldPosition.x;
                    float wz = worldPosition.z;
                    
                    // Complex wave function
                    // Large swells
                    float w1 = sin(wx * 0.05 + time * 0.5) * 0.5;
                    float w2 = cos(wz * 0.05 + time * 0.4) * 0.5;
                    
                    // Detail waves
                    float w3 = sin(wx * 0.2 + wz * 0.1 + time * 1.5) * 0.2;
                    float w4 = cos(wz * 0.2 - wx * 0.1 + time * 1.2) * 0.2;
                    
                    // Choppy surface
                    float w5 = sin((wx + wz) * 0.5 + time * 3.0) * 0.1;
                    
                    float totalWave = w1 + w2 + w3 + w4 + w5;
                    
                    pos.z += totalWave; // Move vertex up/down in local space
                    vWave = totalWave;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_Position = projectionMatrix * mvPosition;
                    
                    #include <logdepthbuf_vertex>
                    #include <fog_vertex>
                }
            `,
            fragmentShader: `
                #include <common>
                #include <fog_pars_fragment>
                #include <logdepthbuf_pars_fragment>
                
                uniform vec3 waterColor;
                uniform vec3 foamColor;
                uniform float alpha;
                uniform float time;
                
                varying vec2 vUv;
                varying float vWave;
                
                void main() {
                    #include <logdepthbuf_fragment>
                    
                    vec3 color = waterColor;
                    
                    // Foam on wave peaks
                    float foam = smoothstep(0.6, 1.2, vWave);
                    color = mix(color, foamColor, foam * 0.5);
                    
                    // Fake specular/sparkle
                    float sparkleTime = time * 2.0;
                    float sparkle = sin(vUv.x * 100.0 + sparkleTime) * cos(vUv.y * 100.0 + sparkleTime);
                    sparkle = pow(max(0.0, sparkle), 20.0); // Sharp highlights
                    
                    color += vec3(sparkle) * 0.3;
                    
                    // Depth cue (darker in valleys of waves)
                    color *= 0.8 + 0.2 * smoothstep(-1.0, 1.0, vWave);
                    
                    gl_FragColor = vec4(color, alpha);

                    #include <fog_fragment>
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false, // Helps with transparency sorting often
            fog: true
        });
    }
}

// ===== CHUNK SYSTEM =====
class TerrainChunk {
    constructor(group, x, z, size, resolution, generator, material, waterMaterial, treePrototypes, bushPrototypes, branchPrototypes, stonePrototypes, logPrototype, vegMaterials, mushroomPrototypes) {
        this.group = group;
        this.x = x;
        this.z = z;
        this.size = size;
        this.resolution = resolution;
        this.generator = generator;
        this.material = material;
        this.waterMaterial = waterMaterial;
        this.treePrototypes = treePrototypes;
        this.bushPrototypes = bushPrototypes;
        this.branchPrototypes = branchPrototypes;
        this.stonePrototypes = stonePrototypes;
        this.mushroomPrototypes = mushroomPrototypes;
        this.logPrototype = logPrototype;
        this.vegMaterials = vegMaterials;

        this.branchMeshes = [];
        this.bushMeshes = [];
        this.stoneMeshes = [];
        this.logMeshes = [];
        this.treeWoodMeshes = [];
        this.treeLeafMeshes = [];
        this.mushroomMeshes = [];
        this.mesh = null;
        this.waterMesh = null;
        this.instancedMeshes = [];
        this.heightField = null;

        this.build();
    }

    build() {
        const geometry = new THREE.PlaneGeometry(this.size, this.size, this.resolution, this.resolution);
        const positions = geometry.attributes.position.array;
        const normals = geometry.attributes.normal.array;
        this.heightField = new Float32Array((this.resolution + 1) * (this.resolution + 1));

        const treeData = [];
        const bushData = [];
        const stoneData = [];

        for (let i = 0; i < positions.length / 3; i++) {
            const xi = i % (this.resolution + 1);
            const yi = Math.floor(i / (this.resolution + 1));

            const localX = (xi / this.resolution - 0.5) * this.size;
            // Plane Y+ (Top, yi=0) corresponds to World Z- (North).
            // So yi=0 should map to negative Z. 
            // -0.5 * size.
            // (0 - 0.5) * size = -0.5 * size.
            // So we need POSITIVE factor.
            const localZ = (yi / this.resolution - 0.5) * this.size;

            const wx = this.x + localX;
            const wz = this.z + localZ;

            const h = this.generator.getElevation(wx, wz);

            positions[i * 3 + 2] = h;
            this.heightField[i] = h;

            // Seamless Normals
            const n = this.generator.getNormal(wx, wz);
            // Transform normal to geometry space (Plane Z is up). 
            // Our normal n.y is World Up.
            // PlaneGeometry is created on XY plane? No default is XY plane, Z up.
            // We rotate mesh X -90 later.
            // So Geometry Z becomes World Y (Up).
            // So we map: n.y (Up) -> z.
            // n.x (Right) -> x.
            // n.z (Forward) -> -y (since we rotate Z->-Y... wait).
            // Let's assume standard mapping:
            // n.x -> x
            // n.y -> z (Up in Geometry)
            // n.z -> -y (Forward in World is -Z in Geometry if rotated? No.)
            // Let's stick to: n.x, n.z is planar, n.y is scaling factor?
            // Actually, keep it simple: Map world normal to local normal.
            // Mesh Rotation X -90:
            // Local (x, y, z) -> World (x, -z, y)
            // We want World (nx, ny, nz)
            // So: x = nx
            // -z = nz => z = -nz
            // y = ny
            // So Local Normal = (nx, ny, -nz) assuming Y is Up in local? 
            // BUT PlaneGeometry Z is Up. So Up (ny) should map to Z.
            // So Local = (nx, -nz, ny).

            normals[i * 3 + 0] = n.x;
            normals[i * 3 + 1] = -n.z;
            normals[i * 3 + 2] = n.y;

            // Vegetation and debris probability
            const vegRand = Math.random();
            const stoneRand = Math.random();

            if (this.treePrototypes && vegRand > 0.99) {
                if (h > CONFIG.terrain.waterLevel + 10 && h < CONFIG.terrain.maxHeight * 0.8) {
                    if (n.y > 0.85) { // Only on flatter ground
                        treeData.push({ x: localX, y: h, z: localZ });
                    }
                }
            } else if (this.bushPrototypes && vegRand > 0.96) {
                if (h > CONFIG.terrain.waterLevel + 2 && h < CONFIG.terrain.maxHeight * 0.9) {
                    if (n.y > 0.75) {
                        bushData.push({ x: localX, y: h, z: localZ });
                    }
                }
            }

            // Stones check (independent of trees/bushes)
            if (stoneRand > 0.97) {
                if (h > CONFIG.terrain.waterLevel + 1) {
                    stoneData.push({ x: localX, y: h, z: localZ });
                }
            }
        }

        const mushroomData = [];
        // Generate mushrooms around trees (clusters)
        if (this.mushroomPrototypes && treeData.length > 0) {
            treeData.forEach(tree => {
                const numMushrooms = Math.floor(Math.random() * 4);
                for (let m = 0; m < numMushrooms; m++) {
                    const angle = Math.random() * Math.PI * 2;
                    const r = 2 + Math.random() * 4;
                    const mx = tree.x + Math.cos(angle) * r;
                    const mz = tree.z + Math.sin(angle) * r;
                    const mh = this.generator.getElevation(this.x + mx, this.z + mz);
                    if (mh > CONFIG.terrain.waterLevel + 2) {
                        mushroomData.push({ x: mx, y: mh, z: mz, type: Math.random() > 0.5 ? 'red' : 'yellow' });
                    }
                }
            });
        }
        // Also some random scattered ones
        for (let i = 0; i < 15; i++) {
            const mx = (Math.random() - 0.5) * this.size;
            const mz = (Math.random() - 0.5) * this.size;
            const mh = this.generator.getElevation(this.x + mx, this.z + mz);
            if (mh > CONFIG.terrain.waterLevel + 2) {
                mushroomData.push({ x: mx, y: mh, z: mz, type: Math.random() > 0.5 ? 'red' : 'yellow' });
            }
        }

        // geometry.computeVertexNormals(); // Disable automatic normals

        this.mesh = new THREE.Mesh(geometry, this.material);
        this.mesh.rotation.x = -Math.PI / 2;
        this.mesh.position.set(this.x, 0, this.z);
        this.mesh.receiveShadow = true;
        this.mesh.castShadow = true;

        // Water
        const waterGeo = new THREE.PlaneGeometry(this.size, this.size, 16, 16);
        this.waterMesh = new THREE.Mesh(waterGeo, this.waterMaterial);
        this.waterMesh.rotation.x = -Math.PI / 2;
        this.waterMesh.position.set(this.x, CONFIG.terrain.waterLevel, this.z);

        this.group.add(this.mesh);
        this.group.add(this.waterMesh);

        if (treeData.length > 0) this.generateTrees(treeData);
        if (bushData.length > 0) this.generateBushes(bushData);
        if (stoneData.length > 0) this.generateStones(stoneData);
        if (mushroomData.length > 0) this.generateMushrooms(mushroomData);

        // Generate debris around trees
        if (this.branchPrototypes && treeData.length > 0) {
            const branchData = [];
            treeData.forEach(tree => {
                // 1-3 branches near each tree
                const numBranches = 1 + Math.floor(Math.random() * 3);
                for (let b = 0; b < numBranches; b++) {
                    const angle = Math.random() * Math.PI * 2;
                    const r = 5 + Math.random() * 15;
                    const bx = tree.x + Math.cos(angle) * r;
                    const bz = tree.z + Math.sin(angle) * r;
                    const bh = this.generator.getElevation(this.x + bx, this.z + bz);

                    if (bh > CONFIG.terrain.waterLevel + 2) {
                        branchData.push({ x: bx, y: bh, z: bz });
                    }
                }
            });
            if (branchData.length > 0) this.generateFallenBranches(branchData);
        }
    }

    generateTrees(treeData) {
        if (!this.treePrototypes || this.treePrototypes.length === 0) return;

        // Group trees by prototype index
        const clusters = Array(this.treePrototypes.length).fill().map(() => []);

        treeData.forEach(data => {
            const protoIdx = Math.floor(Math.random() * this.treePrototypes.length);
            clusters[protoIdx].push(data);
        });

        const dummy = new THREE.Object3D();

        // Create meshes for each prototype
        this.treePrototypes.forEach((proto, idx) => {
            const instances = clusters[idx];
            if (instances.length === 0) return;

            const woodMesh = new THREE.InstancedMesh(proto.wood, this.vegMaterials.wood, instances.length);
            const leafMesh = new THREE.InstancedMesh(proto.leaves, this.vegMaterials.leaf, instances.length);

            woodMesh.castShadow = true;
            woodMesh.receiveShadow = true;
            leafMesh.castShadow = true;
            leafMesh.receiveShadow = true;

            instances.forEach((data, i) => {
                // data.y is height. Trees start at base 0 (relative to their origin).
                // data.x (localX), data.z (localZ).
                // World Pos = this.x + data.x, data.y, this.z + data.z

                dummy.position.set(this.x + data.x, data.y, this.z + data.z);

                const scale = 0.8 + Math.random() * 0.6;
                dummy.scale.set(scale, scale, scale);

                dummy.rotation.set(0, Math.random() * Math.PI * 2, 0);
                dummy.updateMatrix();

                woodMesh.setMatrixAt(i, dummy.matrix);
                leafMesh.setMatrixAt(i, dummy.matrix);
            });

            woodMesh.instanceMatrix.needsUpdate = true;
            leafMesh.instanceMatrix.needsUpdate = true;

            woodMesh.instanceMatrix.needsUpdate = true;
            leafMesh.instanceMatrix.needsUpdate = true;

            woodMesh.userData.isTree = true;
            woodMesh.userData.leafMesh = leafMesh;
            woodMesh.userData.protoIdx = idx;

            this.group.add(woodMesh);
            this.group.add(leafMesh);

            this.instancedMeshes.push(woodMesh, leafMesh);
            this.treeWoodMeshes.push(woodMesh);
            this.treeLeafMeshes.push(leafMesh);
        });
    }

    generateBushes(bushData) {
        if (!this.bushPrototypes || this.bushPrototypes.length === 0) return;

        const clusters = Array(this.bushPrototypes.length).fill().map(() => []);
        bushData.forEach(data => {
            const protoIdx = Math.floor(Math.random() * this.bushPrototypes.length);
            clusters[protoIdx].push(data);
        });

        const dummy = new THREE.Object3D();

        this.bushPrototypes.forEach((proto, idx) => {
            const instances = clusters[idx];
            if (instances.length === 0) return;

            const leafMesh = new THREE.InstancedMesh(proto.leaves, this.vegMaterials.bushLeaf, instances.length);
            const berryMesh = new THREE.InstancedMesh(proto.berries, this.vegMaterials.berry, instances.length);

            leafMesh.castShadow = true;
            leafMesh.receiveShadow = true;
            berryMesh.castShadow = true;

            instances.forEach((data, i) => {
                dummy.position.set(this.x + data.x, data.y, this.z + data.z);
                const scale = 0.8 + Math.random() * 0.4;
                dummy.scale.set(scale, scale, scale);
                dummy.rotation.set(0, Math.random() * Math.PI * 2, 0);
                dummy.updateMatrix();

                leafMesh.setMatrixAt(i, dummy.matrix);
                berryMesh.setMatrixAt(i, dummy.matrix);
            });

            leafMesh.instanceMatrix.needsUpdate = true;
            berryMesh.instanceMatrix.needsUpdate = true;

            // Link them for interaction
            leafMesh.userData.isBush = true;
            leafMesh.userData.berryMesh = berryMesh;

            this.group.add(leafMesh);
            this.group.add(berryMesh);
            this.instancedMeshes.push(leafMesh, berryMesh);
            this.bushMeshes.push(leafMesh);
        });
    }

    generateFallenBranches(branchData) {
        if (!this.branchPrototypes || this.branchPrototypes.length === 0) return;

        const clusters = Array(this.branchPrototypes.length).fill().map(() => []);
        branchData.forEach(data => {
            const protoIdx = Math.floor(Math.random() * this.branchPrototypes.length);
            clusters[protoIdx].push(data);
        });

        const dummy = new THREE.Object3D();

        this.branchPrototypes.forEach((proto, idx) => {
            const instances = clusters[idx];
            if (instances.length === 0) return;

            const mesh = new THREE.InstancedMesh(proto, this.vegMaterials.log, instances.length);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            instances.forEach((data, i) => {
                dummy.position.set(this.x + data.x, data.y, this.z + data.z);
                const scale = 0.5 + Math.random() * 0.5;
                dummy.scale.set(scale, scale, scale);
                dummy.rotation.set(
                    (Math.random() - 0.5) * 0.5,
                    Math.random() * Math.PI * 2,
                    (Math.random() - 0.5) * 0.5
                );
                dummy.updateMatrix();
                mesh.setMatrixAt(i, dummy.matrix);
            });

            mesh.instanceMatrix.needsUpdate = true;
            mesh.userData.isBranch = true;
            this.group.add(mesh);
            this.instancedMeshes.push(mesh);
            this.branchMeshes.push(mesh);
        });
    }

    generateStones(stoneData) {
        if (!this.stonePrototypes || this.stonePrototypes.length === 0) return;

        const stonePrototypes = this.stonePrototypes;
        const clusters = Array(stonePrototypes.length).fill().map(() => []);
        stoneData.forEach(data => {
            const protoIdx = Math.floor(Math.random() * stonePrototypes.length);
            clusters[protoIdx].push(data);
        });

        const dummy = new THREE.Object3D();

        stonePrototypes.forEach((proto, idx) => {
            const instances = clusters[idx];
            if (instances.length === 0) return;

            const mesh = new THREE.InstancedMesh(proto, this.vegMaterials.stone, instances.length);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.userData.isStone = true;

            instances.forEach((data, i) => {
                dummy.position.set(this.x + data.x, data.y, this.z + data.z);
                dummy.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
                const scale = (0.4 + Math.random() * 0.8) * 1.5; // Stones 50% bigger
                dummy.scale.set(scale, scale, scale);
                dummy.updateMatrix();
                mesh.setMatrixAt(i, dummy.matrix);
            });

            mesh.instanceMatrix.needsUpdate = true;
            this.group.add(mesh);
            this.instancedMeshes.push(mesh);
            this.stoneMeshes.push(mesh);
        });
    }

    generateLogs(logData) {
        if (!this.logPrototype) return;

        const mesh = new THREE.InstancedMesh(this.logPrototype, this.vegMaterials.log, logData.length);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        const dummy = new THREE.Object3D();

        logData.forEach((data, i) => {
            const worldX = this.x + data.x;
            const worldZ = this.z + data.z;
            const groundH = this.generator.getElevation(worldX, worldZ);
            const normal = this.generator.getNormal(worldX, worldZ);

            // Position: Ground height + radius (0.9)
            dummy.position.set(worldX, groundH + 0.9, worldZ);

            // Rotation Logic:
            // 1. Start with laying them flat on the ground (rotating around X or Z)
            const baseQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI * 0.5);
            // 2. Give them a random spin around the vertical axis
            const yawQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.random() * Math.PI * 2);
            // 3. Align them to the surface normal
            const slopeQuat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);

            // Combine: Align to slope -> apply random rotation -> lay flat
            dummy.quaternion.copy(slopeQuat).multiply(yawQuat).multiply(baseQuat);

            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
        });

        mesh.userData.isLog = true;
        this.group.add(mesh);
        this.instancedMeshes.push(mesh);
        this.logMeshes.push(mesh);
    }

    generateMushrooms(mushroomData) {
        if (!this.mushroomPrototypes) return;

        const redData = mushroomData.filter(m => m.type === 'red');
        const yellowData = mushroomData.filter(m => m.type === 'yellow');

        const dummy = new THREE.Object3D();

        const createMushroomPlacements = (data, color) => {
            const stemMesh = new THREE.InstancedMesh(this.mushroomPrototypes[color].stem, this.vegMaterials.mushroomStem, data.length);
            const capMesh = new THREE.InstancedMesh(this.mushroomPrototypes[color].cap, color === 'red' ? this.vegMaterials.mushroomRed : this.vegMaterials.mushroomYellow, data.length);

            stemMesh.castShadow = true;
            capMesh.castShadow = true;

            data.forEach((m, i) => {
                dummy.position.set(this.x + m.x, m.y, this.z + m.z);
                dummy.rotation.set(0, Math.random() * Math.PI * 2, 0);
                const s = 0.8 + Math.random() * 0.4;
                dummy.scale.set(s, s, s);
                dummy.updateMatrix();
                stemMesh.setMatrixAt(i, dummy.matrix);
                capMesh.setMatrixAt(i, dummy.matrix);
            });

            stemMesh.instanceMatrix.needsUpdate = true;
            capMesh.instanceMatrix.needsUpdate = true;

            stemMesh.userData.isMushroom = true;
            stemMesh.userData.mushroomType = color;
            stemMesh.userData.capMesh = capMesh;

            this.group.add(stemMesh, capMesh);
            this.instancedMeshes.push(stemMesh, capMesh);
            this.mushroomMeshes.push(stemMesh);
        };

        if (redData.length > 0) createMushroomPlacements(redData, 'red');
        if (yellowData.length > 0) createMushroomPlacements(yellowData, 'yellow');
    }

    sampleHeightAtWorld(worldX, worldZ) {
        if (!this.heightField) return null;

        const u = (worldX - this.x) / this.size + 0.5;
        const v = (worldZ - this.z) / this.size + 0.5;
        if (u < 0 || u > 1 || v < 0 || v > 1) return null;

        const r = this.resolution;
        const eps = 1e-6;
        const gx = Math.min(Math.max(u * r, 0), r - eps);
        const gz = Math.min(Math.max(v * r, 0), r - eps);

        const ix = Math.floor(gx);
        const iz = Math.floor(gz);
        const fx = gx - ix;
        const fz = gz - iz;

        const row = r + 1;
        const i00 = iz * row + ix;
        const i10 = i00 + 1;
        const i01 = i00 + row;
        const i11 = i01 + 1;

        const h00 = this.heightField[i00];
        const h10 = this.heightField[i10];
        const h01 = this.heightField[i01];
        const h11 = this.heightField[i11];

        // Match PlaneGeometry triangulation: (00,01,10) and (01,11,10).
        if (fx + fz <= 1.0) {
            return h00 + fx * (h10 - h00) + fz * (h01 - h00);
        }

        const ax = 1.0 - fx;
        const az = 1.0 - fz;
        return h11 + ax * (h01 - h11) + az * (h10 - h11);
    }

    dispose() {
        if (this.mesh) {
            this.group.remove(this.mesh);
            this.mesh.geometry.dispose();
        }
        if (this.waterMesh) {
            this.group.remove(this.waterMesh);
            this.waterMesh.geometry.dispose();
        }

        this.instancedMeshes.forEach(mesh => {
            this.group.remove(mesh);
            mesh.dispose(); // Only dispose mesh/geometry, NOT the shared materials
        });
        this.instancedMeshes = [];
        this.heightField = null;
    }
}

// ===== FREE FLIGHT CAMERA CONTROLLER =====
class FreeFlyCamera {
    constructor(camera, domElement, generator) {
        this.camera = camera;
        this.domElement = domElement;
        this.generator = generator;

        this.moveState = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            sprint: false,
        };

        this.euler = new THREE.Euler(0, 0, 0, 'YXZ');
        this.velocity = new THREE.Vector3();
        this.verticalVelocity = 0;
        this.locked = false;

        this.mode = 'freefly'; // 'freefly' or 'walk'
        this.playerHeight = 5.0;
        this.gravity = 25.0;
        this.jumpForce = 12.0;
        this.isGrounded = false;

        this.init();
    }

    init() {
        document.addEventListener('click', () => {
            if (!this.locked) {
                this.domElement.requestPointerLock();
            }
        });

        document.addEventListener('pointerlockchange', () => {
            this.locked = document.pointerLockElement === this.domElement;
        });

        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));
    }

    onMouseMove(event) {
        if (!this.locked) return;

        const { movementX, movementY } = event;

        this.euler.setFromQuaternion(this.camera.quaternion);
        this.euler.y -= movementX * CONFIG.camera.mouseSensitivity;
        this.euler.x -= movementY * CONFIG.camera.mouseSensitivity;
        this.euler.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.euler.x));

        this.camera.quaternion.setFromEuler(this.euler);
    }

    onKeyDown(event) {
        switch (event.code) {
            case 'KeyW': this.moveState.forward = true; break;
            case 'KeyS': this.moveState.backward = true; break;
            case 'KeyA': this.moveState.left = true; break;
            case 'KeyD': this.moveState.right = true; break;
            case 'Space': this.moveState.up = true; event.preventDefault(); break;
            case 'ShiftLeft': this.moveState.down = true; break;
            case 'ControlLeft': this.moveState.sprint = true; break;
        }
    }

    onKeyUp(event) {
        switch (event.code) {
            case 'KeyW': this.moveState.forward = false; break;
            case 'KeyS': this.moveState.backward = false; break;
            case 'KeyA': this.moveState.left = false; break;
            case 'KeyD': this.moveState.right = false; break;
            case 'Space': this.moveState.up = false; break;
            case 'ShiftLeft': this.moveState.down = false; break;
            case 'ControlLeft': this.moveState.sprint = false; break;
        }
    }

    update(delta) {
        if (!this.locked) return;

        const speed = (this.mode === 'walk' ? 15.0 : CONFIG.camera.moveSpeed) * (this.moveState.sprint ? CONFIG.camera.sprintMultiplier : 1);

        if (this.mode === 'freefly') {
            this.velocity.set(0, 0, 0);
            const forward = new THREE.Vector3();
            this.camera.getWorldDirection(forward);
            const right = new THREE.Vector3();
            right.crossVectors(forward, this.camera.up).normalize();

            if (this.moveState.forward) this.velocity.add(forward);
            if (this.moveState.backward) this.velocity.sub(forward);
            if (this.moveState.right) this.velocity.add(right);
            if (this.moveState.left) this.velocity.sub(right);
            if (this.moveState.up) this.velocity.y += 1;
            if (this.moveState.down) this.velocity.y -= 1;

            this.velocity.normalize().multiplyScalar(speed);
            this.camera.position.add(this.velocity.multiplyScalar(delta * 60)); // Standardize to 60fps delta
        } else {
            // WALKING MODE
            const forward = new THREE.Vector3();
            this.camera.getWorldDirection(forward);
            forward.y = 0;
            forward.normalize();

            const right = new THREE.Vector3();
            right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

            const moveDir = new THREE.Vector3(0, 0, 0);
            if (this.moveState.forward) moveDir.add(forward);
            if (this.moveState.backward) moveDir.sub(forward);
            if (this.moveState.right) moveDir.add(right);
            if (this.moveState.left) moveDir.sub(right);

            if (moveDir.lengthSq() > 0) {
                moveDir.normalize();
                this.camera.position.add(moveDir.multiplyScalar(speed * delta));
            }

            // Gravity & Jump
            this.verticalVelocity -= this.gravity * delta;
            this.camera.position.y += this.verticalVelocity * delta;

            const groundH = this.generator.getElevation(this.camera.position.x, this.camera.position.z);
            const minHeight = groundH + this.playerHeight;

            if (this.camera.position.y <= minHeight) {
                this.camera.position.y = minHeight;
                this.verticalVelocity = 0;
                this.isGrounded = true;
            } else {
                this.isGrounded = false;
            }

            if (this.moveState.up && this.isGrounded) {
                this.verticalVelocity = this.jumpForce;
                this.isGrounded = false;
            }
        }
    }
}

// ===== VEGETATION SYSTEM =====
class TreeGenerator {
    constructor() {
        this.woodMaterial = new THREE.MeshStandardMaterial({
            color: 0x5d4037,
            roughness: 0.9,
            flatShading: true
        });

        this.leafMaterial = new THREE.MeshStandardMaterial({
            color: 0x2d5a27,
            roughness: 0.8,
            side: THREE.DoubleSide,
            flatShading: true
        });
    }

    generateTreePrototypes(count = 5) {
        const prototypes = [];
        for (let i = 0; i < count; i++) {
            prototypes.push(this.createProceduralTree(i));
        }
        return prototypes;
    }

    createProceduralTree(seed) {
        const branchGeometries = [];
        const leafGeometries = [];

        // Recursive branch generation
        const generateBranch = (startPoint, direction, length, radius, depth) => {
            if (depth === 0) {
                // Generate fractal leaf at tip
                this.generateLeaf(startPoint, direction, leafGeometries);
                return;
            }

            const endPoint = new THREE.Vector3().copy(startPoint).add(direction.clone().multiplyScalar(length));

            // Create branch segment - Cylinder is Y-aligned by default
            const geometry = new THREE.CylinderGeometry(radius * 0.7, radius, length, 5);
            // No rotation needed for base geometry, aligned to Y

            // Align cylinder to direction
            const quaternion = new THREE.Quaternion();
            quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
            geometry.applyQuaternion(quaternion);

            // Position
            const midPoint = new THREE.Vector3().lerpVectors(startPoint, endPoint, 0.5);
            geometry.translate(midPoint.x, midPoint.y, midPoint.z);

            branchGeometries.push(geometry);

            // Spawn child branches
            const numBranches = 2 + Math.floor(Math.random() * 2);
            for (let i = 0; i < numBranches; i++) {
                const angleX = (Math.random() - 0.5) * 1.5;
                const angleY = (Math.random() - 0.5) * 1.5;
                const angleZ = (Math.random() - 0.5) * 1.5;

                const newDir = direction.clone().applyEuler(new THREE.Euler(angleX, angleY, angleZ)).normalize();
                // Bias upwards (Y-axis)
                newDir.y += 0.5;
                newDir.normalize();

                generateBranch(
                    endPoint,
                    newDir,
                    length * 0.7,
                    radius * 0.7,
                    depth - 1
                );
            }
        };

        // Start trunk - Grow Up relative to tree geometry (Y-axis)
        generateBranch(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0), 15 + Math.random() * 10, 1.5, 4);

        // Merge geometries
        const mergedWood = BufferGeometryUtils.mergeGeometries(branchGeometries);
        const mergedLeaves = BufferGeometryUtils.mergeGeometries(leafGeometries);

        return { wood: mergedWood, leaves: mergedLeaves };
    }

    generateLeaf(position, direction, geometries) {
        // Fractal-like leaf cluster
        const leafSize = 4;
        const geometry = new THREE.ConeGeometry(leafSize, leafSize * 2, 4);
        // Cone is Y-aligned by default

        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
        geometry.applyQuaternion(quaternion);
        geometry.translate(position.x, position.y, position.z);

        geometries.push(geometry);

        // Add smaller sub-leaves for fractal look
        for (let i = 0; i < 3; i++) {
            const subGeo = new THREE.ConeGeometry(leafSize * 0.5, leafSize, 4);
            // By default Y-aligned

            // Random orientation around main direction
            const subDir = direction.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), (i / 3) * Math.PI * 2);
            subDir.add(new THREE.Vector3((Math.random() - 0.5), (Math.random() - 0.5), (Math.random() - 0.5)).multiplyScalar(0.5)).normalize();

            const subQ = new THREE.Quaternion();
            subQ.setFromUnitVectors(new THREE.Vector3(0, 1, 0), subDir);
            subGeo.applyQuaternion(subQ);

            // Offset slightly
            subGeo.translate(position.x + subDir.x * 2, position.y + subDir.y * 2, position.z + subDir.z * 2);

            geometries.push(subGeo);
        }
    }
    generateBushPrototypes(count = 3) {
        const prototypes = [];
        for (let i = 0; i < count; i++) {
            prototypes.push(this.createProceduralBush(i));
        }
        return prototypes;
    }

    createProceduralBush(seed) {
        const leafGeometries = [];
        const berryGeometries = [];

        // Main cluster of leaves
        const numClusters = 5 + Math.floor(Math.random() * 4);
        for (let i = 0; i < numClusters; i++) {
            const size = 3 + Math.random() * 2;
            const geo = new THREE.IcosahedronGeometry(size, 1); // Low poly organic look

            // Random offset from center
            const offset = new THREE.Vector3(
                (Math.random() - 0.5) * 6,
                Math.random() * 4,
                (Math.random() - 0.5) * 6
            );
            geo.translate(offset.x, offset.y, offset.z);
            leafGeometries.push(geo);

            // Add berries on the surface of this leaf cluster
            const numBerries = 3 + Math.floor(Math.random() * 3);
            for (let b = 0; b < numBerries; b++) {
                const berrySize = 0.4 + Math.random() * 0.3;
                const berryGeo = new THREE.SphereGeometry(berrySize, 6, 6);

                // Position berry randomly on a sphere surface slightly larger than the leaf cluster
                const phi = Math.random() * Math.PI * 2;
                const theta = Math.random() * Math.PI;
                const r = size * 0.9;

                const bx = offset.x + r * Math.sin(theta) * Math.cos(phi);
                const by = offset.y + r * Math.sin(theta) * Math.sin(phi);
                const bz = offset.z + r * Math.cos(theta);

                // Only keep berries that are above some height (not buried)
                if (by > 1.0) {
                    berryGeo.translate(bx, by, bz);
                    berryGeometries.push(berryGeo);
                }
            }
        }

        const mergedLeaves = BufferGeometryUtils.mergeGeometries(leafGeometries);
        const mergedBerries = BufferGeometryUtils.mergeGeometries(berryGeometries);

        return { leaves: mergedLeaves, berries: mergedBerries };
    }

    generateFallenBranchPrototypes(count = 3) {
        const prototypes = [];
        for (let i = 0; i < count; i++) {
            prototypes.push(this.createFallenBranch(i));
        }
        return prototypes;
    }

    generateMushroomPrototypes() {
        // Just one prototype per color is enough for now, or we can vary them
        return {
            red: this.createProceduralMushroom(0xff0000),
            yellow: this.createProceduralMushroom(0xffff00)
        };
    }

    createProceduralMushroom() {
        const stemGeo = new THREE.CylinderGeometry(0.15, 0.22, 0.6, 6);
        stemGeo.translate(0, 0.3, 0);

        const capGeo = new THREE.SphereGeometry(0.6, 8, 8, 0, Math.PI * 2, 0, Math.PI * 0.5);
        capGeo.scale(1, 0.6, 1);
        capGeo.translate(0, 0.5, 0);

        return { stem: stemGeo, cap: capGeo };
    }

    createFallenBranch(seed) {
        const geometries = [];
        const numSegments = 2 + Math.floor(Math.random() * 3);
        let start = new THREE.Vector3(0, 0, 0);
        let dir = new THREE.Vector3(1, 0, 0); // Laying on XZ plane

        for (let i = 0; i < numSegments; i++) {
            const length = 2 + Math.random() * 4;
            const radius = 0.2 + (numSegments - i) * 0.1;
            const geo = new THREE.CylinderGeometry(radius * 0.7, radius, length, 4);

            // Align to dir
            const q = new THREE.Quaternion();
            q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            geo.applyQuaternion(q);

            const end = start.clone().add(dir.clone().multiplyScalar(length));
            const mid = start.clone().lerp(end, 0.5);
            geo.translate(mid.x, mid.y, mid.z);

            geometries.push(geo);

            // New start and Slightly random dir
            start = end;
            dir.applyAxisAngle(new THREE.Vector3(0, 1, 0), (Math.random() - 0.5) * 1.5);
            dir.y += (Math.random() - 0.5) * 0.2; // Minor Y variation
            dir.normalize();
        }

        return BufferGeometryUtils.mergeGeometries(geometries);
    }

    generateStonePrototypes(count = 3) {
        const prototypes = [];
        for (let i = 0; i < count; i++) {
            prototypes.push(this.createProceduralStone(i));
        }
        return prototypes;
    }

    createProceduralStone(seed) {
        const size = (0.5 + Math.random() * 1.5) * 1.5; // Base size also 50% bigger
        let geometry = new THREE.IcosahedronGeometry(size, 1); // Detail 1 for more vertices to make "holes"

        // Merge vertices so shared vertices move together during displacement
        geometry = BufferGeometryUtils.mergeVertices(geometry);

        // Randomly squish and deform
        const posAttr = geometry.attributes.position;
        const v = new THREE.Vector3();

        // Use a consistent squish for the whole stone
        const squishX = 0.8 + Math.random() * 0.4;
        const squishY = 0.4 + Math.random() * 0.4;
        const squishZ = 0.8 + Math.random() * 0.4;

        for (let i = 0; i < posAttr.count; i++) {
            v.fromBufferAttribute(posAttr, i);

            // Apply squish
            v.x *= squishX;
            v.y *= squishY;
            v.z *= squishZ;

            // Add slight per-vertex noise
            v.x += (Math.random() - 0.5) * 0.2;
            v.y += (Math.random() - 0.5) * 0.2;
            v.z += (Math.random() - 0.5) * 0.2;

            // Create "holes" or deep pits
            // If a random check passes, we push the vertex significantly towards the center
            if (Math.random() < 0.15) {
                const depth = 0.3 + Math.random() * 0.4;
                v.multiplyScalar(1.0 - depth);
            }

            posAttr.setXYZ(i, v.x, v.y, v.z);
        }

        geometry.computeVertexNormals(); // Ensure smooth but rugged look
        return geometry;
    }

    generateLogPrototype() {
        const geometry = new THREE.CylinderGeometry(0.9, 0.9, 6, 8);
        return geometry;
    }
}

// ===== RAIN SYSTEM =====
class SplashSystem {
    constructor(scene) {
        this.scene = scene;
        this.count = 1000;
        this.duration = 0.5; // seconds
        this.dummy = new THREE.Object3D();
        this.index = 0;

        // Geometry: simple plane
        const geometry = new THREE.PlaneGeometry(5, 5);

        const material = new THREE.ShaderMaterial({
            transparent: true,
            depthWrite: false,
            uniforms: THREE.UniformsUtils.merge([
                THREE.UniformsLib.fog,
                {
                    uTime: { value: 0 },
                    uDuration: { value: this.duration },
                    uColor: { value: new THREE.Color(0xaaaaaa) }
                }
            ]),
            vertexShader: `
                attribute mat4 instanceMatrix;
                uniform float uTime;
                uniform float uDuration;
                attribute float aStartTime;
                varying float vAlpha;
                varying vec2 vUv;
                
                #include <common>
                #include <fog_pars_vertex>

                void main() {
                    vUv = uv;
                    float age = uTime - aStartTime;
                    float progress = age / uDuration;
                    
                    vAlpha = 1.0 - progress;
                    
                    if (progress < 0.0 || progress > 1.0) {
                        progress = 0.0;
                        vAlpha = 0.0; // Hide
                    }
                    
                    float scale = 1.0 + progress * 2.0;
                    
                    vec3 transformed = position * scale;
                    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(transformed, 1.0);
                    gl_Position = projectionMatrix * mvPosition;

                    #include <fog_vertex>
                }
            `,
            fragmentShader: `
                #include <common>
                #include <fog_pars_fragment>

                varying float vAlpha;
                varying vec2 vUv;
                uniform vec3 uColor;
                
                void main() {
                    if (vAlpha <= 0.0) discard;
                    
                    // Ring pattern
                    float dist = distance(vUv, vec2(0.5));
                    float ring = smoothstep(0.3, 0.4, dist) - smoothstep(0.4, 0.5, dist);
                    
                    gl_FragColor = vec4(uColor, ring * vAlpha * 0.5);

                    #include <fog_fragment>
                }
            `,
            fog: true
        });

        this.mesh = new THREE.InstancedMesh(geometry, material, this.count);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        // Keep mesh at identity, handle all positioning/rotation in the instance matrix
        this.scene.add(this.mesh);

        // Custom attribute for start time
        this.startTimes = new Float32Array(this.count);
        this.mesh.geometry.setAttribute('aStartTime', new THREE.InstancedBufferAttribute(this.startTimes, 1));
    }

    spawn(x, z, time) {
        const i = this.index;

        this.dummy.position.set(x, CONFIG.terrain.waterLevel + 0.5, z);
        this.dummy.rotation.x = -Math.PI / 2;
        this.dummy.updateMatrix();

        this.mesh.setMatrixAt(i, this.dummy.matrix);
        this.startTimes[i] = time;

        this.mesh.geometry.attributes.aStartTime.setX(i, time);
        this.mesh.geometry.attributes.aStartTime.needsUpdate = true;
        this.mesh.instanceMatrix.needsUpdate = true;

        this.index = (this.index + 1) % this.count;
    }

    update(time) {
        this.mesh.material.uniforms.uTime.value = time;
    }
}

class RainEffect {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.enabled = false;
        this.count = 20000;
        this.range = 2000;
        this.geometry = new THREE.BufferGeometry();
        this.particles = null;
        this.velocities = [];
        this.splashSystem = null;

        this.init();
    }

    init() {
        const positions = [];

        for (let i = 0; i < this.count; i++) {
            positions.push(
                (Math.random() - 0.5) * this.range, // x
                (Math.random() - 0.5) * this.range, // y
                (Math.random() - 0.5) * this.range  // z
            );
            this.velocities.push(- (30 + Math.random() * 40));
        }

        this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

        const material = new THREE.PointsMaterial({
            color: 0xaaaaaa,
            size: 0.8,
            transparent: true,
            opacity: 0.6,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.particles = new THREE.Points(this.geometry, material);
    }

    setSplashSystem(system) {
        this.splashSystem = system;
    }

    enable() {
        if (this.enabled) return;
        this.enabled = true;
        this.scene.add(this.particles);
        // scene.fog density could be increased here for atmosphere
    }

    disable() {
        if (!this.enabled) return;
        this.enabled = false;
        this.scene.remove(this.particles);
    }

    update(delta, totalTime) {
        if (!this.enabled) return;

        const positions = this.geometry.attributes.position.array;
        const camPos = this.camera.position;
        const halfRange = this.range / 2;
        const waterLevel = CONFIG.terrain.waterLevel;

        for (let i = 0; i < this.count; i++) {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            const v = this.velocities[i];
            const oldY = y;

            // Gravity
            y += v * delta * 5.0;

            // Splash check
            if (this.splashSystem && oldY >= waterLevel && y < waterLevel) {
                // Spawns if near enough to camera to be seen
                const dx = x - camPos.x;
                const dz = z - camPos.z;
                if (dx * dx + dz * dz < 1000 * 1000) {
                    if (Math.random() > 0.5) { // 50% chance
                        this.splashSystem.spawn(x, z, totalTime);
                    }
                }
            }

            // Wrap logic relative to CAMERA
            // Use wide Y range to ensure it hits water 
            if (y < camPos.y - 1500) {
                y = camPos.y + 1500;
                x = camPos.x + (Math.random() - 0.5) * this.range;
                z = camPos.z + (Math.random() - 0.5) * this.range;
            }
            if (y > camPos.y + 1500) {
                y = camPos.y - 1500;
            }

            if (x < camPos.x - halfRange) x += this.range;
            else if (x > camPos.x + halfRange) x -= this.range;

            if (z < camPos.z - halfRange) z += this.range;
            else if (z > camPos.z + halfRange) z -= this.range;

            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;
        }

        this.geometry.attributes.position.needsUpdate = true;
    }
}

// ===== ANIMAL SYSTEM =====
class Pig {
    constructor(scene, generator, startPos, world = null) {
        this.scene = scene;
        this.generator = generator;
        this.world = world;
        this.pingMaterial = new THREE.MeshStandardMaterial({ color: 0xffc0cb, roughness: 0.8, flatShading: true });

        this.group = new THREE.Group();
        this.group.position.copy(startPos);
        this.group.scale.set(3, 3, 3); // 3x larger pigs

        // Body
        this.body = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.8, 2.0), this.pingMaterial);
        this.body.position.y = 0.6;
        this.group.add(this.body);

        // Head
        this.head = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.7, 0.7), this.pingMaterial);
        this.head.position.set(0, 0.8, 1.2);
        this.group.add(this.head);

        // Snout
        const snout = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.3, 0.2), new THREE.MeshStandardMaterial({ color: 0xffa0af, flatShading: true }));
        snout.position.set(0, 0, 0.45);
        this.head.add(snout);

        // Legs
        this.legs = [];
        const legGeo = new THREE.BoxGeometry(0.3, 0.6, 0.3);
        this.legOffsets = [
            [-0.4, 0, 0.7], [0.4, 0, 0.7],
            [-0.4, 0, -0.7], [0.4, 0, -0.7]
        ];

        this.legOffsets.forEach(pos => {
            const leg = new THREE.Mesh(legGeo, this.pingMaterial);
            leg.position.set(pos[0], 0.3, pos[2]);
            this.legs.push(leg);
            this.group.add(leg);
        });

        this.scene.add(this.group);

        this.targetAngle = Math.random() * Math.PI * 2;
        this.angle = this.targetAngle;
        this.speed = 4.0 + Math.random() * 3.0; // Scaled speed
        this.walkCycle = 0;

        this.legHalfHeight = 0.3;
        this.legHalfWidth = 0.15;
        this.legBaseY = 0.3;
        this.footClearance = 0.01;
        this.footProbeLocal = [
            // Point-contact hoof probe to avoid over-lifting from adjacent higher triangles.
            new THREE.Vector3(0, -this.legHalfHeight, 0)
        ];

        this.footGroundHeights = [0, 0, 0, 0];
        this.footPoints = [
            new THREE.Vector3(),
            new THREE.Vector3(),
            new THREE.Vector3(),
            new THREE.Vector3()
        ];

        this.upAxis = new THREE.Vector3(0, 1, 0);
        this.yawQuat = new THREE.Quaternion();
        this.slopeQuat = new THREE.Quaternion();
        this.groundNormal = new THREE.Vector3(0, 1, 0);

        this.tmpMove = new THREE.Vector3();
        this.tmpA = new THREE.Vector3();
        this.tmpB = new THREE.Vector3();
        this.tmpC = new THREE.Vector3();
        this.tmpD = new THREE.Vector3();
        this.tmpWorldUp = new THREE.Vector3();
        this.tmpFootWorld = new THREE.Vector3();

        this.maxHp = 10;
        this.hp = this.maxHp;
        this.dead = false;
        this.createHealthbar();
    }

    createHealthbar() {
        this.healthCanvas = document.createElement('canvas');
        this.healthCanvas.width = 128;
        this.healthCanvas.height = 24;
        this.healthCtx = this.healthCanvas.getContext('2d');
        this.healthTexture = new THREE.CanvasTexture(this.healthCanvas);

        const healthMaterial = new THREE.SpriteMaterial({
            map: this.healthTexture,
            transparent: true,
            depthTest: false,
            depthWrite: false
        });

        this.healthbar = new THREE.Sprite(healthMaterial);
        this.healthbar.scale.set(4.5, 0.8, 1);
        this.healthbar.renderOrder = 5000;
        this.healthbar.visible = false;
        this.scene.add(this.healthbar);

        this.redrawHealthbar();
    }

    redrawHealthbar() {
        const ctx = this.healthCtx;
        if (!ctx) return;

        const w = this.healthCanvas.width;
        const h = this.healthCanvas.height;
        ctx.clearRect(0, 0, w, h);

        if (this.hp <= 0 || this.hp >= this.maxHp) {
            this.healthbar.visible = false;
            this.healthTexture.needsUpdate = true;
            return;
        }

        const pad = 2;
        const innerX = pad + 1;
        const innerY = pad + 1;
        const innerW = w - 2 * (pad + 1);
        const innerH = h - 2 * (pad + 1);

        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        ctx.fillRect(0, 0, w, h);

        ctx.fillStyle = '#2b2b2b';
        ctx.fillRect(innerX, innerY, innerW, innerH);

        const ratio = THREE.MathUtils.clamp(this.hp / this.maxHp, 0, 1);
        const fillW = Math.floor(innerW * ratio);
        ctx.fillStyle = '#e53935';
        ctx.fillRect(innerX, innerY, fillW, innerH);

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.strokeRect(1, 1, w - 2, h - 2);

        this.healthbar.visible = true;
        this.healthTexture.needsUpdate = true;
    }

    takeDamage(amount) {
        if (this.dead) return false;
        this.hp = Math.max(0, this.hp - amount);
        this.redrawHealthbar();

        if (this.hp <= 0) {
            this.die();
            return true;
        }

        return false;
    }

    die() {
        if (this.dead) return;
        this.dead = true;
        const deathPos = this.group.position.clone();

        if (this.world && typeof this.world.onPigKilled === 'function') {
            this.world.onPigKilled(this, deathPos);
        } else {
            this.dispose();
        }
    }

    dispose() {
        this.scene.remove(this.group);
        if (this.healthbar) {
            this.scene.remove(this.healthbar);
            if (this.healthbar.material && this.healthbar.material.map) {
                this.healthbar.material.map.dispose();
            }
            if (this.healthbar.material) this.healthbar.material.dispose();
        }
    }

    getGroundY(x, z) {
        if (this.world && typeof this.world.getTerrainHeightAtPosition === 'function') {
            return this.world.getTerrainHeightAtPosition(x, z);
        }
        return this.generator.getElevation(x, z);
    }

    sampleFootGround(yawAngle) {
        const sx = this.group.scale.x;
        const sz = this.group.scale.z;
        const px = this.group.position.x;
        const pz = this.group.position.z;
        const c = Math.cos(yawAngle);
        const s = Math.sin(yawAngle);

        for (let i = 0; i < this.legOffsets.length; i++) {
            const offset = this.legOffsets[i];
            const point = this.footPoints[i];

            const lx = offset[0] * sx;
            const lz = offset[2] * sz;
            const legWorldX = px + lx * c + lz * s;
            const legWorldZ = pz - lx * s + lz * c;
            const groundY = this.getGroundY(legWorldX, legWorldZ);

            this.footGroundHeights[i] = groundY;
            point.set(legWorldX, groundY, legWorldZ);
        }
    }

    update(delta, time, camera) {
        if (this.dead) return;

        // AI: Gentle wandering
        if (Math.random() < 0.005) {
            this.targetAngle += (Math.random() - 0.5) * 2;
        }

        this.angle += (this.targetAngle - this.angle) * 0.03;

        // Move forward locally
        this.tmpMove.set(0, 0, 1).applyAxisAngle(this.upAxis, this.angle);
        this.group.position.addScaledVector(this.tmpMove, this.speed * delta);

        // Improved Water Avoidance
        const centerH = this.getGroundY(this.group.position.x, this.group.position.z);
        if (centerH < CONFIG.terrain.waterLevel + 4.0) {
            this.targetAngle = this.angle + Math.PI;
            this.tmpMove.set(0, 0, -1).applyAxisAngle(this.upAxis, this.angle);
            this.group.position.addScaledVector(this.tmpMove, this.speed * delta * 2);
        }

        this.sampleFootGround(this.angle);

        // Align body to local slope from leg contact points.
        const p0 = this.footPoints[0];
        const p1 = this.footPoints[1];
        const p2 = this.footPoints[2];
        const p3 = this.footPoints[3];
        const n1 = this.tmpA.subVectors(p1, p0).cross(this.tmpB.subVectors(p2, p0));
        const n2 = this.tmpC.subVectors(p3, p1).cross(this.tmpD.subVectors(p2, p1));
        this.groundNormal.copy(n1).add(n2);
        if (this.groundNormal.lengthSq() < 1e-6) {
            this.groundNormal.copy(this.generator.getNormal(this.group.position.x, this.group.position.z));
        }
        this.groundNormal.normalize();
        if (this.groundNormal.y < 0) this.groundNormal.multiplyScalar(-1);

        this.yawQuat.setFromAxisAngle(this.upAxis, this.angle);
        this.slopeQuat.setFromUnitVectors(this.upAxis, this.groundNormal);
        this.group.quaternion.multiplyQuaternions(this.slopeQuat, this.yawQuat);

        const avgFootH = (this.footGroundHeights[0] + this.footGroundHeights[1] + this.footGroundHeights[2] + this.footGroundHeights[3]) * 0.25;
        this.group.position.y = avgFootH + this.footClearance;

        this.walkCycle += delta * (this.speed / 3.0) * 4.0;
        this.legs.forEach((leg, i) => {
            const phase = (i === 0 || i === 3) ? 0 : Math.PI;
            const walkRot = Math.sin(this.walkCycle + phase) * 0.25;
            leg.rotation.x = walkRot;
            leg.position.set(this.legOffsets[i][0], this.legBaseY, this.legOffsets[i][2]);
        });

        // Solve each leg against exact vertical probe target on the terrain.
        for (let iter = 0; iter < 2; iter++) {
            this.group.updateMatrixWorld(true);
            this.tmpWorldUp.set(0, 1, 0).applyQuaternion(this.group.quaternion);
            const localToWorldY = Math.max(1e-4, this.group.scale.y * this.tmpWorldUp.y);

            for (let i = 0; i < this.legs.length; i++) {
                const leg = this.legs[i];
                this.tmpFootWorld.set(0, -this.legHalfHeight, 0);
                leg.localToWorld(this.tmpFootWorld);

                const targetY = this.footGroundHeights[i] + this.footClearance;
                const deltaLocalY = (targetY - this.tmpFootWorld.y) / localToWorldY;
                leg.position.y += deltaLocalY;
            }
        }

        // Sync body bounce
        this.body.position.y = 0.6 + Math.abs(Math.sin(this.walkCycle)) * 0.1; // More bounce for larger body

        if (this.healthbar) {
            this.healthbar.position.set(this.group.position.x, this.group.position.y + 5.2, this.group.position.z);
            this.healthbar.visible = this.hp > 0 && this.hp < this.maxHp;
        }
    }
}

class Rabbit {
    constructor(scene, generator, startPos, world = null) {
        this.scene = scene;
        this.generator = generator;
        this.world = world;
        this.material = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.9, flatShading: true });

        this.group = new THREE.Group();
        this.group.position.copy(startPos);
        this.group.scale.set(0.8, 0.8, 0.8); // Standard rabbit size

        // Body
        this.body = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.5, 0.8), this.material);
        this.body.position.y = 0.4;
        this.group.add(this.body);

        // Head
        this.head = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.4), this.material);
        this.head.position.set(0, 0.6, 0.4);
        this.group.add(this.head);

        // Ears
        const earGeo = new THREE.BoxGeometry(0.1, 0.5, 0.05);
        const leftEar = new THREE.Mesh(earGeo, this.material);
        leftEar.position.set(-0.1, 0.4, -0.1);
        leftEar.rotation.x = -0.2;
        this.head.add(leftEar);

        const rightEar = new THREE.Mesh(earGeo, this.material);
        rightEar.position.set(0.1, 0.4, -0.1);
        rightEar.rotation.x = -0.2;
        this.head.add(rightEar);

        // Tail
        const tail = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.2, 0.2), this.material);
        tail.position.set(0, 0, -0.4);
        this.body.add(tail);

        // Legs
        this.legs = [];
        this.legOffsets = [
            [-0.2, -0.2, 0.3], [0.2, -0.2, 0.3],
            [-0.2, -0.2, -0.3], [0.2, -0.2, -0.3]
        ];
        const legGeo = new THREE.BoxGeometry(0.15, 0.3, 0.15);
        this.legOffsets.forEach(pos => {
            const leg = new THREE.Mesh(legGeo, this.material);
            leg.position.set(pos[0], pos[1], pos[2]);
            this.legs.push(leg);
            this.body.add(leg);
        });

        this.scene.add(this.group);

        this.targetAngle = Math.random() * Math.PI * 2;
        this.angle = this.targetAngle;

        // AI States
        this.state = 'idle'; // 'idle' or 'hopping'
        this.stateTimer = Math.random() * 2;
        this.hopProgress = 0;
        this.hopPower = 0.4;
        this.hopLength = 3.0;

        this.footGroundHeights = [0, 0, 0, 0];
        this.footPoints = [
            new THREE.Vector3(),
            new THREE.Vector3(),
            new THREE.Vector3(),
            new THREE.Vector3()
        ];
        this.legHalfHeight = 0.15;
        this.footClearance = 0.01;

        this.upAxis = new THREE.Vector3(0, 1, 0);
        this.yawQuat = new THREE.Quaternion();
        this.slopeQuat = new THREE.Quaternion();
        this.groundNormal = new THREE.Vector3(0, 1, 0);

        this.tmpMove = new THREE.Vector3();
        this.tmpA = new THREE.Vector3();
        this.tmpB = new THREE.Vector3();
        this.tmpC = new THREE.Vector3();
        this.tmpD = new THREE.Vector3();
        this.tmpFootWorld = new THREE.Vector3();
        this.tmpWorldUp = new THREE.Vector3();

        this.baseFootOffset = 0.4 + (-0.2) - this.legHalfHeight;
    }

    getGroundY(x, z) {
        if (this.world && typeof this.world.getTerrainHeightAtPosition === 'function') {
            return this.world.getTerrainHeightAtPosition(x, z);
        }
        return this.generator.getElevation(x, z);
    }

    sampleFootGround(yawAngle) {
        const sx = this.group.scale.x;
        const sz = this.group.scale.z;
        const px = this.group.position.x;
        const pz = this.group.position.z;
        const c = Math.cos(yawAngle);
        const s = Math.sin(yawAngle);

        for (let i = 0; i < this.legOffsets.length; i++) {
            const off = this.legOffsets[i];
            const lx = off[0] * sx;
            const lz = off[2] * sz;
            const wx = px + lx * c + lz * s;
            const wz = pz - lx * s + lz * c;
            const h = this.getGroundY(wx, wz);

            this.footGroundHeights[i] = h;
            this.footPoints[i].set(wx, h, wz);
        }

        const p0 = this.footPoints[0];
        const p1 = this.footPoints[1];
        const p2 = this.footPoints[2];
        const p3 = this.footPoints[3];
        const n1 = this.tmpA.subVectors(p1, p0).cross(this.tmpB.subVectors(p2, p0));
        const n2 = this.tmpC.subVectors(p3, p1).cross(this.tmpD.subVectors(p2, p1));

        this.groundNormal.copy(n1).add(n2);
        if (this.groundNormal.lengthSq() < 1e-6) {
            this.groundNormal.copy(this.generator.getNormal(px, pz));
        }
        this.groundNormal.normalize();
        if (this.groundNormal.y < 0) this.groundNormal.multiplyScalar(-1);
    }

    update(delta, time) {
        this.stateTimer -= delta;

        if (this.state === 'idle') {
            if (this.stateTimer <= 0) {
                this.state = 'hopping';
                this.stateTimer = 0.5 + Math.random() * 0.5; // Hop duration
                this.hopProgress = 0;
                // Randomly turn while idle
                if (Math.random() < 0.4) {
                    this.targetAngle += (Math.random() - 0.5) * 4;
                }
            }
        } else if (this.state === 'hopping') {
            this.hopProgress += delta * 2.5;

            // Move forward
            const moveSpeed = this.hopLength;
            this.tmpMove.set(0, 0, 1).applyAxisAngle(this.upAxis, this.angle);
            this.group.position.addScaledVector(this.tmpMove, moveSpeed * delta);

            // Jump arc
            this.body.position.y = 0.4 + Math.sin(this.hopProgress * Math.PI) * 0.8;

            // Leg tuck
            this.legs.forEach(leg => {
                leg.rotation.x = Math.sin(this.hopProgress * Math.PI) * 0.5;
            });

            if (this.hopProgress >= 1.0) {
                this.state = 'idle';
                this.stateTimer = 1.0 + Math.random() * 3.0; // Wait before next hop
                this.body.position.y = 0.4;
                this.legs.forEach(leg => leg.rotation.x = 0);
            }
        }

        this.angle += (this.targetAngle - this.angle) * 0.1;

        // Ground/Water Logic
        const h = this.getGroundY(this.group.position.x, this.group.position.z);

        if (h < CONFIG.terrain.waterLevel + 1.0) {
            this.targetAngle = this.angle + Math.PI;
            // Immediate nudge out of water
            this.tmpMove.set(0, 0, -1).applyAxisAngle(this.upAxis, this.angle);
            this.group.position.addScaledVector(this.tmpMove, 5.0 * delta);
        }

        this.sampleFootGround(this.angle);

        const avgFootH = (this.footGroundHeights[0] + this.footGroundHeights[1] + this.footGroundHeights[2] + this.footGroundHeights[3]) * 0.25;
        this.group.position.y = avgFootH + this.footClearance - (this.baseFootOffset * this.group.scale.y);

        this.yawQuat.setFromAxisAngle(this.upAxis, this.angle);
        this.slopeQuat.setFromUnitVectors(this.upAxis, this.groundNormal);
        this.group.quaternion.multiplyQuaternions(this.slopeQuat, this.yawQuat);

        // Solve each leg against exact vertical probe target on the terrain.
        for (let iter = 0; iter < 2; iter++) {
            this.group.updateMatrixWorld(true);
            this.tmpWorldUp.set(0, 1, 0).applyQuaternion(this.group.quaternion);
            const localToWorldY = Math.max(1e-4, this.group.scale.y * this.tmpWorldUp.y);

            for (let i = 0; i < this.legs.length; i++) {
                const leg = this.legs[i];
                this.tmpFootWorld.set(0, -this.legHalfHeight, 0);
                leg.localToWorld(this.tmpFootWorld);

                const targetY = this.footGroundHeights[i] + this.footClearance;
                const deltaLocalY = (targetY - this.tmpFootWorld.y) / localToWorldY;
                leg.position.y += deltaLocalY;
            }
        }
    }
}

class Firepit {
    constructor(scene, pos) {
        this.scene = scene;
        this.group = new THREE.Group();
        this.group.position.copy(pos);

        // Logs in a pile
        const logGeo = new THREE.CylinderGeometry(0.3, 0.3, 2.4, 8);
        const logMat = new THREE.MeshStandardMaterial({ color: 0x3d2b1f, roughness: 0.9, flatShading: true });

        for (let i = 0; i < 5; i++) {
            const log = new THREE.Mesh(logGeo, logMat);
            log.rotation.z = Math.PI / 2;
            log.rotation.y = (i / 5) * Math.PI * 2;
            log.position.y = 0.3;
            this.group.add(log);
        }

        // Fire animation
        const fireGeo = new THREE.ConeGeometry(0.9, 2.4, 6);
        const fireLightMat = new THREE.MeshBasicMaterial({ color: 0xff6600, transparent: true, opacity: 0.7 });
        this.fire = new THREE.Mesh(fireGeo, fireLightMat);
        this.fire.position.y = 1.35; // Slightly above logs
        this.group.add(this.fire);

        // Light
        this.light = new THREE.PointLight(0xffa500, 80, 60);
        this.light.position.y = 2.5;
        this.group.add(this.light);

        this.scene.add(this.group);
        this.time = Math.random() * 10;
    }

    update(delta) {
        this.time += delta;
        const scale = 0.8 + Math.sin(this.time * 12) * 0.2;
        this.fire.scale.set(scale, scale * (1.2 + Math.cos(this.time * 18) * 0.4), scale);
        this.fire.rotation.y += delta * 4;
        this.light.intensity = 60 + Math.sin(this.time * 15) * 30;
    }
}

class Wall {
    constructor(scene, pos, rotation) {
        this.scene = scene;
        this.width = 5.0;
        this.height = 5.0; // Square
        this.group = new THREE.Group();
        this.group.position.copy(pos);
        this.group.rotation.y = rotation;

        // Stacked logs look
        const logGeo = new THREE.CylinderGeometry(0.25, 0.25, this.width, 8);
        const logMat = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 0.9, flatShading: true });

        const numLogs = 10; // 10 * 0.5 = 5.0 height
        for (let i = 0; i < numLogs; i++) {
            const log = new THREE.Mesh(logGeo, logMat);
            log.rotation.z = Math.PI / 2;
            log.position.y = 0.25 + i * 0.5;
            this.group.add(log);
        }

        this.scene.add(this.group);

        // Snap points in world space
        // Snap points in world space
        this.snapPoints = [
            { pos: new THREE.Vector3(-this.width / 2, 0, 0), type: 'bottom_left' },
            { pos: new THREE.Vector3(this.width / 2, 0, 0), type: 'bottom_right' },
            { pos: new THREE.Vector3(0, 0, 0), type: 'bottom_center' },
            { pos: new THREE.Vector3(0, this.height, 0), type: 'top_center' },
            { pos: new THREE.Vector3(-this.width / 2, this.height, 0), type: 'top_left' },
            { pos: new THREE.Vector3(this.width / 2, this.height, 0), type: 'top_right' }
        ];
    }

    getSnapPoints() {
        return this.snapPoints.map(sp => ({
            pos: sp.pos.clone().applyMatrix4(this.group.matrixWorld),
            type: sp.type,
            rotation: this.group.rotation.y
        }));
    }
}

class Floor {
    constructor(scene, pos, rotation) {
        this.scene = scene;
        this.size = 5.0;
        this.group = new THREE.Group();
        this.group.position.copy(pos);
        this.group.rotation.y = rotation;

        // Planks look
        const plankGeo = new THREE.BoxGeometry(0.48, 0.1, this.size);
        const plankMat = new THREE.MeshStandardMaterial({ color: 0x795548, roughness: 0.8, flatShading: true });

        for (let i = 0; i < 10; i++) {
            const plank = new THREE.Mesh(plankGeo, plankMat);
            plank.position.x = -this.size / 2 + 0.25 + i * 0.5;
            this.group.add(plank);
        }

        this.scene.add(this.group);

        // Snap points
        this.snapPoints = [
            { pos: new THREE.Vector3(-this.size / 2, 0, 0), type: 'floor_edge' },
            { pos: new THREE.Vector3(this.size / 2, 0, 0), type: 'floor_edge' },
            { pos: new THREE.Vector3(0, 0, -this.size / 2), type: 'floor_edge' },
            { pos: new THREE.Vector3(0, 0, this.size / 2), type: 'floor_edge' },
            // Corners
            { pos: new THREE.Vector3(-this.size / 2, 0, -this.size / 2), type: 'floor_corner' },
            { pos: new THREE.Vector3(this.size / 2, 0, -this.size / 2), type: 'floor_corner' },
            { pos: new THREE.Vector3(-this.size / 2, 0, this.size / 2), type: 'floor_corner' },
            { pos: new THREE.Vector3(this.size / 2, 0, this.size / 2), type: 'floor_corner' }
        ];
    }

    getSnapPoints() {
        return this.snapPoints.map(sp => ({
            pos: sp.pos.clone().applyMatrix4(this.group.matrixWorld),
            type: sp.type,
            rotation: this.group.rotation.y
        }));
    }
}

class Roof {
    constructor(scene, pos, rotation, thatchMat) {
        this.scene = scene;
        this.width = 5.0;
        this.length = 5.0 * Math.sqrt(2); // Spans exactly one floor/wall unit horizontally/vertically at 45 deg
        this.angle = Math.PI / 4; // 45 degrees
        this.group = new THREE.Group();
        this.group.position.copy(pos);
        this.group.rotation.y = rotation;

        // Sloped planks look
        const slopeGroup = new THREE.Group();
        slopeGroup.rotation.x = this.angle; // Sloping UPWARD
        this.group.add(slopeGroup);

        const plankGeo = new THREE.BoxGeometry(0.48, 0.1, this.length);
        const plankMat = new THREE.MeshStandardMaterial({ color: 0x4e342e, roughness: 0.8, flatShading: true });

        for (let i = 0; i < 10; i++) {
            const plank = new THREE.Mesh(plankGeo, plankMat);
            plank.position.x = -this.width / 2 + 0.25 + i * 0.5;
            plank.position.z = -this.length / 2;
            slopeGroup.add(plank);
        }

        // Thatch layer on the OUTSIDE (above the planks)
        const thatchGeo = new THREE.BoxGeometry(this.width, 0.15, this.length);
        const thatch = new THREE.Mesh(thatchGeo, thatchMat || new THREE.MeshStandardMaterial({ color: 0xd4c28d }));
        thatch.position.y = 0.1; // Slightly above the 0.1 thick planks
        thatch.position.z = -this.length / 2;
        slopeGroup.add(thatch);

        this.scene.add(this.group);

        // Snap points
        this.snapPoints = [
            { pos: new THREE.Vector3(0, 0, 0), type: 'roof_bottom_center' },
            { pos: new THREE.Vector3(0, this.length * Math.sin(this.angle), -this.length * Math.cos(this.angle)), type: 'roof_top_center' },
            { pos: new THREE.Vector3(-this.width / 2, (this.length / 2) * Math.sin(this.angle), -(this.length / 2) * Math.cos(this.angle)), type: 'roof_side' },
            { pos: new THREE.Vector3(this.width / 2, (this.length / 2) * Math.sin(this.angle), -(this.length / 2) * Math.cos(this.angle)), type: 'roof_side' },
            // Corner points
            { pos: new THREE.Vector3(-this.width / 2, 0, 0), type: 'roof_bottom_corner' },
            { pos: new THREE.Vector3(this.width / 2, 0, 0), type: 'roof_bottom_corner' },
            { pos: new THREE.Vector3(-this.width / 2, this.length * Math.sin(this.angle), -this.length * Math.cos(this.angle)), type: 'roof_top_corner' },
            { pos: new THREE.Vector3(this.width / 2, this.length * Math.sin(this.angle), -this.length * Math.cos(this.angle)), type: 'roof_top_corner' }
        ];
    }

    getSnapPoints() {
        return this.snapPoints.map(sp => ({
            pos: sp.pos.clone().applyMatrix4(this.group.matrixWorld),
            type: sp.type,
            rotation: this.group.rotation.y
        }));
    }
}

class WallTriangle {
    constructor(scene, pos, rotation, flipped = false) {
        this.scene = scene;
        this.width = 5.0;
        this.height = 5.0;
        this.group = new THREE.Group();
        this.group.position.copy(pos);
        this.group.rotation.y = rotation;
        if (flipped) this.group.scale.x = -1;

        // Stacked logs look (triangle)
        // We use 10 logs like the regular wall
        const numLogs = 10;
        const logMat = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 0.9, flatShading: true });

        for (let i = 0; i < numLogs; i++) {
            // Logs get shorter as we go up
            // i=0 (bottom) -> width=5.0
            // i=9 (top) -> width=0.5
            const currentWidth = this.width * (1 - (i / numLogs));
            const logGeo = new THREE.CylinderGeometry(0.25, 0.25, currentWidth, 8);
            const log = new THREE.Mesh(logGeo, logMat);
            log.rotation.z = Math.PI / 2;
            log.position.y = 0.25 + i * 0.5;
            // Shift log so the right-angled side is at X=0
            log.position.x = -currentWidth / 2;
            this.group.add(log);
        }

        this.scene.add(this.group);

        // Snap points
        this.snapPoints = [
            { pos: new THREE.Vector3(0, 0, 0), type: 'tri_corner_90' },
            { pos: new THREE.Vector3(-this.width, 0, 0), type: 'tri_corner_thin' },
        ];
    }

    getSnapPoints() {
        return this.snapPoints.map(sp => ({
            pos: sp.pos.clone().applyMatrix4(this.group.matrixWorld),
            type: sp.type,
            rotation: this.group.rotation.y
        }));
    }
}

class VolumetricCloudSystem {
    constructor(scene) {
        this.scene = scene;
        this.group = new THREE.Group();
        this.group.renderOrder = 1;
        this.scene.add(this.group);

        this.enabled = true;
        this.time = 0;
        this.bounds = 16000;
        this.clouds = [];

        this.createClouds();
    }

    createClouds() {
        const cloudMaterial = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.45,
            roughness: 1.0,
            metalness: 0.0,
            depthWrite: false
        });

        const baseSphere = new THREE.SphereGeometry(1, 10, 8);
        const cloudCount = 36;

        for (let i = 0; i < cloudCount; i++) {
            const puffCount = 6 + Math.floor(Math.random() * 6);
            const puffGeometries = [];

            for (let p = 0; p < puffCount; p++) {
                const puff = baseSphere.clone();
                const matrix = new THREE.Matrix4();
                const px = (Math.random() - 0.5) * 180;
                const py = (Math.random() - 0.5) * 45;
                const pz = (Math.random() - 0.5) * 120;
                const s = 22 + Math.random() * 42;

                matrix.compose(
                    new THREE.Vector3(px, py, pz),
                    new THREE.Quaternion(),
                    new THREE.Vector3(s, s * (0.7 + Math.random() * 0.4), s)
                );
                puff.applyMatrix4(matrix);
                puffGeometries.push(puff);
            }

            const merged = BufferGeometryUtils.mergeGeometries(puffGeometries, false);
            puffGeometries.forEach(g => g.dispose());

            const cloud = new THREE.Mesh(merged, cloudMaterial);
            cloud.position.set(
                (Math.random() - 0.5) * this.bounds * 2,
                900 + Math.random() * 900,
                (Math.random() - 0.5) * this.bounds * 2
            );

            const drift = 5 + Math.random() * 10;
            const driftZ = (Math.random() - 0.5) * 2.5;
            const phase = Math.random() * Math.PI * 2;

            this.clouds.push({
                mesh: cloud,
                baseY: cloud.position.y,
                drift,
                driftZ,
                phase
            });
            this.group.add(cloud);
        }

        baseSphere.dispose();
    }

    setEnabled(enabled) {
        this.enabled = enabled;
        this.group.visible = enabled;
    }

    update(delta, cameraPos) {
        if (!this.enabled) return;

        this.time += delta;
        const minX = cameraPos.x - this.bounds;
        const maxX = cameraPos.x + this.bounds;
        const minZ = cameraPos.z - this.bounds;
        const maxZ = cameraPos.z + this.bounds;

        for (const cloud of this.clouds) {
            const mesh = cloud.mesh;
            mesh.position.x += cloud.drift * delta;
            mesh.position.z += cloud.driftZ * delta;
            mesh.position.y = cloud.baseY + Math.sin(this.time * 0.08 + cloud.phase) * 10;

            if (mesh.position.x > maxX) mesh.position.x = minX;
            if (mesh.position.x < minX) mesh.position.x = maxX;
            if (mesh.position.z > maxZ) mesh.position.z = minZ;
            if (mesh.position.z < minZ) mesh.position.z = maxZ;
        }
    }
}

// ===== MAIN SCENE =====
class TerrainScene {
    constructor() {
        this.container = document.getElementById('canvas-container');

        // Update CONFIG for 8x scale (64000)
        CONFIG.terrain.size = 64000;
        CONFIG.camera.far = 25000;

        this.chunkSize = 2000;
        this.chunkResolution = 64;
        this.chunks = new Map(); // "x,z" -> Chunk
        this.viewDistance = 6000;
        this.mistEnabled = false;
        this.windEnabled = false;
        this.cloudsEnabled = true;
        this.windTime = 0;

        this.inventory = { branches: 0, berries: 0, stones: 0, logs: 0, meat: 0, leather: 0, redMushrooms: 0, yellowMushrooms: 0, tools: [] };
        this.nextToolId = 1;
        this.actionbarSlots = new Array(9).fill(null);
        this.activeSlotIndex = 0;
        this.actionbarSlotElements = [];
        this.droppedItems = [];
        this.focusedItem = null; // { mesh, index, type }
        this.craftingOpen = false;
        this.pigs = [];
        this.rabbits = [];
        this.firepits = [];
        this.walls = [];
        this.triWalls = [];
        this.floors = [];
        this.roofs = [];
        this.activeFallingTrees = []; // List of { group, pos, chunk, vel, angle, axis, quat }
        this.placementMode = null; // 'firepit', 'wall', 'floor', 'roof' or null
        this.placementGhost = null;
        this.placementRotation = 0; // 0 to 15 (16 steps)
        this.placementFlipped = false;

        // Shared materials for vegetation with wind support
        this.windUniforms = {
            uWindTime: { value: 0 },
            uWindStrength: { value: 0 }
        };

        this.generator = new TerrainGenerator(CONFIG);
        this.createVegetationMaterials();

        this.setupScene();
        this.setupLighting();
        this.cloudSystem = new VolumetricCloudSystem(this.scene);
        this.cloudSystem.setEnabled(this.cloudsEnabled);

        this.treeGenerator = new TreeGenerator();
        this.treePrototypes = this.treeGenerator.generateTreePrototypes(5); // Generate 5 variations
        this.bushPrototypes = this.treeGenerator.generateBushPrototypes(3); // Generate 3 variations
        this.branchPrototypes = this.treeGenerator.generateFallenBranchPrototypes(3);
        this.stonePrototypes = this.treeGenerator.generateStonePrototypes(3);
        this.logPrototype = this.treeGenerator.generateLogPrototype();
        this.mushroomPrototypes = this.treeGenerator.generateMushroomPrototypes();

        this.terrainGroup = new THREE.Group();
        this.scene.add(this.terrainGroup);

        // Materials (shared)
        this.terrainMaterial = this.generator.createTerrainMaterial();
        this.waterMaterial = this.generator.createWaterMaterial();

        this.rainEffect = new RainEffect(this.scene, this.camera);
        this.splashSystem = new SplashSystem(this.scene);
        this.rainEffect.setSplashSystem(this.splashSystem);

        this.meatDropMaterial = new THREE.MeshStandardMaterial({ color: 0xb04132, roughness: 0.8, flatShading: true });
        this.leatherDropMaterial = new THREE.MeshStandardMaterial({ color: 0x8d6e63, roughness: 0.95, flatShading: true });

        this.setupControls();
        this.setupUI();
        this.createViewmodel();
        this.spawnPigs();
        this.spawnRabbits();
        this.animate();
        this.setupCraftingListeners();
        this.updateInventoryUI();

        // Hide loading screen
        setTimeout(() => {
            const loading = document.getElementById('loading');
            if (loading) loading.style.display = 'none';
            const panel = document.getElementById('info-panel');
            if (panel) panel.style.display = 'block';
        }, 500);
    }

    createVegetationMaterials() {
        const applyWind = (shader) => {
            shader.uniforms.uWindTime = this.windUniforms.uWindTime;
            shader.uniforms.uWindStrength = this.windUniforms.uWindStrength;

            shader.vertexShader = `
                uniform float uWindTime;
                uniform float uWindStrength;
            ` + shader.vertexShader;

            shader.vertexShader = shader.vertexShader.replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>
                // Wind sway: stronger at the top (higher Y)
                // Use instance translation for spatial variance so the whole tree sways together
                vec3 instanceWorldPos = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
                float sway = sin(uWindTime * 1.5 + instanceWorldPos.x * 0.01 + instanceWorldPos.z * 0.01) * 
                             cos(uWindTime * 0.8 + instanceWorldPos.x * 0.02);
                
                // Strength increases with local height
                float swayStrength = uWindStrength * max(0.0, position.y * 0.1);
                
                // For trees/bushes, X/Z sway
                transformed.x += sway * swayStrength;
                transformed.z += sway * swayStrength * 0.5;
                `
            );
        };

        this.woodMaterial = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 0.9, flatShading: true });
        this.woodMaterial.onBeforeCompile = applyWind;

        this.leafMaterial = new THREE.MeshStandardMaterial({ color: 0x2d5a27, roughness: 0.8, side: THREE.DoubleSide, flatShading: true });
        this.leafMaterial.onBeforeCompile = applyWind;

        this.bushLeafMaterial = new THREE.MeshStandardMaterial({ color: 0x1a4a1a, roughness: 0.8, flatShading: true });
        this.bushLeafMaterial.onBeforeCompile = applyWind;

        this.berryMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000, roughness: 0.3, metalness: 0.5 });
        this.berryMaterial.onBeforeCompile = applyWind;

        this.stoneMaterial = new THREE.MeshStandardMaterial({ color: 0x9999aa, roughness: 0.9, flatShading: true });

        // Static material for logs (no wind)
        this.logMaterial = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 0.9, flatShading: true });

        // Mushroom Materials
        this.mushroomStemMaterial = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.9, flatShading: true });
        this.mushroomRedMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000, roughness: 0.8, flatShading: true });
        this.mushroomYellowMaterial = new THREE.MeshStandardMaterial({ color: 0xffcc00, roughness: 0.8, flatShading: true });

        // Thatch for roofs
        this.thatchTexture = this.generator.createTexture('#d4c28d', '#a68b44', 0.8);
        this.thatchMaterial = new THREE.MeshStandardMaterial({
            map: this.thatchTexture,
            roughness: 1.0,
            flatShading: true
        });
    }

    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x87ceeb);
        this.scene.fog = new THREE.Fog(0x87ceeb, this.viewDistance * 0.5, this.viewDistance);

        this.camera = new THREE.PerspectiveCamera(
            CONFIG.camera.fov,
            window.innerWidth / window.innerHeight,
            CONFIG.camera.near,
            CONFIG.camera.far
        );
        this.camera.position.set(0, 800, 0);
        this.scene.add(this.camera); // Add camera to scene so children (viewmodels) render

        this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0.5, 1, 0.5); // Better angle
        this.scene.add(directionalLight);

        const hemisphereLight = new THREE.HemisphereLight(0x87ceeb, 0x6b8e23, 0.4);
        this.scene.add(hemisphereLight);
    }

    getTerrainHeightAtPosition(x, z) {
        const fallback = this.generator.getElevation(x, z);
        const half = this.chunkSize * 0.5;
        const baseX = Math.floor((x + half) / this.chunkSize) * this.chunkSize;
        const baseZ = Math.floor((z + half) / this.chunkSize) * this.chunkSize;

        let chunk = this.chunks.get(`${baseX},${baseZ}`);
        let hitY = chunk ? chunk.sampleHeightAtWorld(x, z) : null;
        if (hitY !== null && hitY !== undefined) return hitY;

        // Edge fallback: try neighboring chunks if primary lookup misses.
        for (let dx = -1; dx <= 1; dx++) {
            for (let dz = -1; dz <= 1; dz++) {
                if (dx === 0 && dz === 0) continue;
                chunk = this.chunks.get(`${baseX + dx * this.chunkSize},${baseZ + dz * this.chunkSize}`);
                if (!chunk) continue;
                hitY = chunk.sampleHeightAtWorld(x, z);
                if (hitY !== null && hitY !== undefined) return hitY;
            }
        }

        return fallback;
    }

    updateChunks() {
        const cx = Math.floor(this.camera.position.x / this.chunkSize) * this.chunkSize;
        const cz = Math.floor(this.camera.position.z / this.chunkSize) * this.chunkSize;

        const chunksToLoad = [];
        const maxDist = this.viewDistance;

        // Identify chunks in range
        for (let x = cx - maxDist; x <= cx + maxDist; x += this.chunkSize) {
            for (let z = cz - maxDist; z <= cz + maxDist; z += this.chunkSize) {
                const dx = x - this.camera.position.x;
                const dz = z - this.camera.position.z;
                if (dx * dx + dz * dz < maxDist * maxDist) {
                    chunksToLoad.push(`${x},${z}`);
                }
            }
        }

        // Remove old chunks
        for (const [key, chunk] of this.chunks) {
            if (!chunksToLoad.includes(key)) {
                chunk.dispose();
                this.chunks.delete(key);
            }
        }

        // Add new chunks
        for (const key of chunksToLoad) {
            if (!this.chunks.has(key)) {
                const [x, z] = key.split(',').map(Number);
                const chunk = new TerrainChunk(
                    this.terrainGroup,
                    x, z,
                    this.chunkSize,
                    this.chunkResolution,
                    this.generator,
                    this.terrainMaterial,
                    this.waterMaterial,
                    this.treePrototypes,
                    this.bushPrototypes,
                    this.branchPrototypes,
                    this.stonePrototypes,
                    this.logPrototype,
                    {
                        wood: this.woodMaterial,
                        leaf: this.leafMaterial,
                        bushLeaf: this.bushLeafMaterial,
                        berry: this.berryMaterial,
                        stone: this.stoneMaterial,
                        log: this.logMaterial,
                        mushroomStem: this.mushroomStemMaterial,
                        mushroomRed: this.mushroomRedMaterial,
                        mushroomYellow: this.mushroomYellowMaterial
                    },
                    this.mushroomPrototypes
                );
                this.chunks.set(key, chunk);
            }
        }
    }

    setupControls() {
        this.controls = new FreeFlyCamera(this.camera, this.renderer.domElement, this.generator);

        window.addEventListener('keydown', (e) => {
            if (e.code === 'KeyE' && this.focusedItem) {
                this.pickupItem();
            }
            if (e.code === 'KeyC') {
                this.toggleCrafting();
            }
            if (e.code.startsWith('Digit') || e.code.startsWith('Numpad')) {
                const slot = parseInt(e.code.replace('Digit', '').replace('Numpad', ''), 10);
                if (slot >= 1 && slot <= 9) {
                    this.activateActionbarSlot(slot - 1);
                }
            }
            if (e.code === 'KeyR' && this.placementMode) {
                if (this.placementMode === 'tri_wall') {
                    this.placementFlipped = !this.placementFlipped;
                } else {
                    this.placementRotation = (this.placementRotation + 1) % 16;
                }
            }
        });

        window.addEventListener('mousedown', (e) => {
            if (e.button === 0 && this.controls.locked && !this.craftingOpen) {
                this.handleAction();
            }
        });

        // Raycaster for interaction
        this.raycaster = new THREE.Raycaster();
        this.raycaster.far = 100; // Max reach

        // Highlight mesh (yellow outline)
        const highlightMat = new THREE.MeshBasicMaterial({ color: 0xffff00, wireframe: true, transparent: true, opacity: 0.5 });
        this.highlightMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), highlightMat);
        this.highlightMesh.visible = false;
        this.scene.add(this.highlightMesh);
    }

    pickupItem() {
        const { mesh, index } = this.focusedItem;
        if (!mesh) return;

        let picked = false;
        let itemName = "";

        if (mesh.userData.isDrop) {
            if (mesh.userData.dropType === 'meat') {
                this.inventory.meat++;
                itemName = "Raw Meat";
            } else if (mesh.userData.dropType === 'leather') {
                this.inventory.leather++;
                itemName = "Leather";
            }

            this.scene.remove(mesh);
            this.droppedItems = this.droppedItems.filter(item => item !== mesh);

            if (mesh.geometry) mesh.geometry.dispose();
            picked = true;
        } else {
            const matrix = new THREE.Matrix4();

            if (mesh.userData.isBranch) {
                mesh.getMatrixAt(index, matrix);
                matrix.scale(new THREE.Vector3(0, 0, 0));
                mesh.setMatrixAt(index, matrix);
                mesh.instanceMatrix.needsUpdate = true;
                this.inventory.branches++;
                picked = true;
                itemName = "Branch";
            } else if (mesh.userData.isBush) {
                const berryMesh = mesh.userData.berryMesh;
                if (berryMesh) {
                    berryMesh.getMatrixAt(index, matrix);
                    // Check if already 0
                    const s = new THREE.Vector3();
                    matrix.decompose(new THREE.Vector3(), new THREE.Quaternion(), s);
                    if (s.length() > 0.001) {
                        matrix.scale(new THREE.Vector3(0, 0, 0));
                        berryMesh.setMatrixAt(index, matrix);
                        berryMesh.instanceMatrix.needsUpdate = true;
                        this.inventory.berries += 5;
                        picked = true;
                        itemName = "Berries";
                    }
                }
            } else if (mesh.userData.isStone) {
                mesh.getMatrixAt(index, matrix);
                matrix.scale(new THREE.Vector3(0, 0, 0));
                mesh.setMatrixAt(index, matrix);
                mesh.instanceMatrix.needsUpdate = true;
                this.inventory.stones++;
                picked = true;
                itemName = "Stone";
            } else if (mesh.userData.isLog) {
                mesh.getMatrixAt(index, matrix);
                matrix.scale(new THREE.Vector3(0, 0, 0));
                mesh.setMatrixAt(index, matrix);
                mesh.instanceMatrix.needsUpdate = true;
                this.inventory.logs++;
                picked = true;
                itemName = "Log";
            } else if (mesh.userData.isMushroom) {
                mesh.getMatrixAt(index, matrix);
                matrix.scale(new THREE.Vector3(0, 0, 0));
                mesh.setMatrixAt(index, matrix);
                mesh.instanceMatrix.needsUpdate = true;

                const capMesh = mesh.userData.capMesh;
                if (capMesh) {
                    capMesh.setMatrixAt(index, matrix);
                    capMesh.instanceMatrix.needsUpdate = true;
                }

                if (mesh.userData.mushroomType === 'red') {
                    this.inventory.redMushrooms++;
                    itemName = "Red Mushroom";
                } else {
                    this.inventory.yellowMushrooms++;
                    itemName = "Yellow Mushroom";
                }
                picked = true;
            }
        }

        if (picked) {
            this.updateInventoryUI();
            this.showPickupFeedback(itemName);
        }

        this.focusedItem = null;
        this.highlightMesh.visible = false;
    }

    spawnPigs() {
        for (let i = 0; i < 40; i++) {
            const x = (Math.random() - 0.5) * 600;
            const z = (Math.random() - 0.5) * 600;
            const h = this.generator.getElevation(x, z);
            if (h > CONFIG.terrain.waterLevel + 2) {
                const pig = new Pig(this.scene, this.generator, new THREE.Vector3(x, h, z), this);
                this.pigs.push(pig);
            }
        }
    }

    spawnRabbits() {
        for (let i = 0; i < 50; i++) {
            const x = (Math.random() - 0.5) * 800;
            const z = (Math.random() - 0.5) * 800;
            const h = this.generator.getElevation(x, z);
            if (h > CONFIG.terrain.waterLevel + 1) {
                const rabbit = new Rabbit(this.scene, this.generator, new THREE.Vector3(x, h, z), this);
                this.rabbits.push(rabbit);
            }
        }
    }

    getToolById(toolId) {
        return this.inventory.tools.find(t => t.id === toolId) || null;
    }

    getActiveTool() {
        const toolId = this.actionbarSlots[this.activeSlotIndex];
        if (toolId === null || toolId === undefined) return null;
        return this.getToolById(toolId);
    }

    isChopTool(tool) {
        return !!tool && tool.type === 'Stone Axe';
    }

    getToolShortName(type) {
        if (type === 'Stone Axe') return 'Axe';
        if (type === 'Stone Club') return 'Club';
        return type;
    }

    activateActionbarSlot(index) {
        if (index < 0 || index >= this.actionbarSlots.length) return;
        this.activeSlotIndex = index;
        this.updateActionbarUI();
        this.updateViewmodel();
    }

    addToolToInventory(type, durability) {
        const tool = { id: this.nextToolId++, type, durability };
        this.inventory.tools.push(tool);

        const emptySlot = this.actionbarSlots.findIndex(slot => slot === null);
        if (emptySlot !== -1) {
            this.actionbarSlots[emptySlot] = tool.id;
            if (!this.getActiveTool()) {
                this.activeSlotIndex = emptySlot;
            }
        }

        return tool;
    }

    removeToolById(toolId) {
        this.inventory.tools = this.inventory.tools.filter(t => t.id !== toolId);
        for (let i = 0; i < this.actionbarSlots.length; i++) {
            if (this.actionbarSlots[i] === toolId) {
                this.actionbarSlots[i] = null;
            }
        }
    }

    setupActionbar() {
        if (document.getElementById('actionbar')) {
            this.updateActionbarUI();
            return;
        }

        const bar = document.createElement('div');
        bar.id = 'actionbar';
        bar.style.position = 'fixed';
        bar.style.left = '50%';
        bar.style.bottom = '20px';
        bar.style.transform = 'translateX(-50%)';
        bar.style.display = 'grid';
        bar.style.gridTemplateColumns = 'repeat(9, 76px)';
        bar.style.gap = '6px';
        bar.style.zIndex = '1200';
        bar.style.pointerEvents = 'auto';

        this.actionbarSlotElements = [];
        for (let i = 0; i < 9; i++) {
            const slot = document.createElement('div');
            slot.dataset.slot = String(i);
            slot.style.width = '76px';
            slot.style.height = '64px';
            slot.style.border = '2px solid rgba(255,255,255,0.35)';
            slot.style.borderRadius = '8px';
            slot.style.background = 'rgba(0,0,0,0.55)';
            slot.style.backdropFilter = 'blur(6px)';
            slot.style.position = 'relative';
            slot.style.userSelect = 'none';
            slot.style.cursor = 'pointer';

            slot.addEventListener('mousedown', (e) => e.stopPropagation());
            slot.addEventListener('click', (e) => {
                e.stopPropagation();
                this.activateActionbarSlot(i);
            });

            slot.draggable = true;
            slot.addEventListener('dragstart', (e) => {
                if (this.actionbarSlots[i] === null) {
                    e.preventDefault();
                    return;
                }
                if (e.dataTransfer) {
                    e.dataTransfer.setData('text/plain', String(i));
                    e.dataTransfer.effectAllowed = 'move';
                }
                slot.style.opacity = '0.55';
            });
            slot.addEventListener('dragend', () => {
                slot.style.opacity = '1';
                this.actionbarSlotElements.forEach(el => { el.style.outline = ''; });
            });
            slot.addEventListener('dragover', (e) => {
                e.preventDefault();
                slot.style.outline = '2px solid rgba(255,255,255,0.7)';
            });
            slot.addEventListener('dragleave', () => {
                slot.style.outline = '';
            });
            slot.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                slot.style.outline = '';

                const fromIndex = parseInt(e.dataTransfer?.getData('text/plain') ?? '', 10);
                if (Number.isNaN(fromIndex) || fromIndex < 0 || fromIndex > 8 || fromIndex === i) return;

                const tmp = this.actionbarSlots[fromIndex];
                this.actionbarSlots[fromIndex] = this.actionbarSlots[i];
                this.actionbarSlots[i] = tmp;

                if (this.activeSlotIndex === fromIndex) this.activeSlotIndex = i;
                else if (this.activeSlotIndex === i) this.activeSlotIndex = fromIndex;

                this.updateActionbarUI();
                this.updateViewmodel();
            });

            bar.appendChild(slot);
            this.actionbarSlotElements.push(slot);
        }

        document.body.appendChild(bar);
        this.updateActionbarUI();
    }

    updateActionbarUI() {
        if (!this.actionbarSlotElements || this.actionbarSlotElements.length === 0) return;

        const validIds = new Set(this.inventory.tools.map(t => t.id));
        for (let i = 0; i < this.actionbarSlots.length; i++) {
            const toolId = this.actionbarSlots[i];
            if (toolId !== null && !validIds.has(toolId)) {
                this.actionbarSlots[i] = null;
            }
        }

        for (let i = 0; i < this.actionbarSlotElements.length; i++) {
            const slot = this.actionbarSlotElements[i];
            const tool = this.getToolById(this.actionbarSlots[i]);
            const selected = i === this.activeSlotIndex;

            slot.style.border = selected ? '2px solid #ffd54f' : '2px solid rgba(255,255,255,0.35)';
            slot.style.boxShadow = selected ? '0 0 10px rgba(255,213,79,0.6)' : 'none';
            slot.innerHTML = '';

            const hotkey = document.createElement('div');
            hotkey.textContent = String(i + 1);
            hotkey.style.position = 'absolute';
            hotkey.style.top = '3px';
            hotkey.style.left = '6px';
            hotkey.style.fontSize = '11px';
            hotkey.style.color = 'rgba(255,255,255,0.85)';
            hotkey.style.fontFamily = 'monospace';
            slot.appendChild(hotkey);

            if (tool) {
                const label = document.createElement('div');
                label.textContent = this.getToolShortName(tool.type);
                label.style.position = 'absolute';
                label.style.left = '50%';
                label.style.top = '22px';
                label.style.transform = 'translateX(-50%)';
                label.style.fontSize = '12px';
                label.style.fontWeight = '600';
                label.style.color = 'white';
                label.style.whiteSpace = 'nowrap';
                slot.appendChild(label);

                const durability = document.createElement('div');
                durability.textContent = `${Math.max(0, Math.round(tool.durability))}%`;
                durability.style.position = 'absolute';
                durability.style.left = '50%';
                durability.style.bottom = '5px';
                durability.style.transform = 'translateX(-50%)';
                durability.style.fontSize = '11px';
                durability.style.color = 'rgba(255,255,255,0.9)';
                slot.appendChild(durability);
            }
        }
    }

    onPigKilled(pig, deathPos) {
        const idx = this.pigs.indexOf(pig);
        if (idx !== -1) {
            this.pigs.splice(idx, 1);
        }

        this.spawnPigDrops(deathPos);
        pig.dispose();
    }

    spawnPigDrops(centerPos) {
        const meatCount = 1 + Math.floor(Math.random() * 3);
        const leatherCount = 1;
        const total = meatCount + leatherCount;

        for (let i = 0; i < total; i++) {
            const isLeather = i >= meatCount;
            const angle = Math.random() * Math.PI * 2;
            const dist = 1.0 + Math.random() * 2.5;
            const x = centerPos.x + Math.cos(angle) * dist;
            const z = centerPos.z + Math.sin(angle) * dist;
            const y = this.generator.getElevation(x, z) + 3.0 + Math.random() * 1.5;
            this.spawnDropItem(isLeather ? 'leather' : 'meat', new THREE.Vector3(x, y, z));
        }
    }

    spawnDropItem(type, pos) {
        // 50% larger drops than the original request baseline.
        const mesh = type === 'leather'
            ? new THREE.Mesh(new THREE.BoxGeometry(0.825, 0.12, 0.6), this.leatherDropMaterial)
            : new THREE.Mesh(new THREE.BoxGeometry(0.57, 0.42, 0.42), this.meatDropMaterial);

        mesh.position.copy(pos);
        mesh.rotation.y = Math.random() * Math.PI * 2;
        if (type === 'leather') {
            // Keep leather flat so corners cannot clip through terrain.
            mesh.rotation.x = 0;
            mesh.rotation.z = 0;
        }
        mesh.castShadow = false;
        mesh.receiveShadow = false;

        mesh.userData.isDrop = true;
        mesh.userData.dropType = type;
        mesh.userData.pickupName = type === 'leather' ? 'Leather' : 'Raw Meat';
        mesh.userData.dropHalfSize = type === 'leather'
            ? new THREE.Vector3(0.4125, 0.06, 0.3)
            : new THREE.Vector3(0.285, 0.21, 0.21);
        mesh.userData.dropVelY = 0;
        mesh.userData.dropGrounded = false;

        const supportY = this.getDropSupportY(mesh);
        if (mesh.position.y < supportY + 1.5) {
            mesh.position.y = supportY + 1.5;
        }

        this.scene.add(mesh);
        this.droppedItems.push(mesh);
    }

    getDropSupportY(mesh) {
        if (!mesh) return 0;
        const half = mesh.userData.dropHalfSize;
        if (!half) {
            const groundY = this.generator.getElevation(mesh.position.x, mesh.position.z);
            return groundY + 0.2;
        }

        const c = Math.cos(mesh.rotation.y);
        const s = Math.sin(mesh.rotation.y);
        let maxGround = -Infinity;
        const probeHalfX = half.x + 0.08;
        const probeHalfZ = half.z + 0.08;

        // Probe dense footprint grid and sit above the highest sampled ground.
        for (let ix = -3; ix <= 3; ix++) {
            for (let iz = -3; iz <= 3; iz++) {
                const ox = (ix / 3) * probeHalfX;
                const oz = (iz / 3) * probeHalfZ;
                // Use the same Y-axis rotation orientation as three.js applyAxisAngle.
                const wx = mesh.position.x + ox * c + oz * s;
                const wz = mesh.position.z - ox * s + oz * c;
                const gh = this.generator.getElevation(wx, wz);
                if (gh > maxGround) maxGround = gh;
            }
        }

        const safetyMargin = 0.12;
        return maxGround + half.y + safetyMargin;
    }

    keepDropAboveGround(mesh) {
        if (!mesh) return;
        mesh.position.y = this.getDropSupportY(mesh);
    }

    updateDroppedItemsPhysics(delta) {
        const gravity = -18.0;
        for (const mesh of this.droppedItems) {
            if (!mesh.userData.dropGrounded) {
                mesh.userData.dropVelY += gravity * delta;
                mesh.position.y += mesh.userData.dropVelY * delta;
            }

            const supportY = this.getDropSupportY(mesh);
            if (mesh.position.y <= supportY) {
                mesh.position.y = supportY;
                mesh.userData.dropVelY = 0;
                mesh.userData.dropGrounded = true;
            }
        }
    }

    tryHitPig(tool) {
        const meleeRange = 14;
        const forward = new THREE.Vector3();
        this.camera.getWorldDirection(forward);

        let bestPig = null;
        let bestDist = Infinity;
        const toPig = new THREE.Vector3();

        for (const pig of this.pigs) {
            if (!pig || pig.dead) continue;

            toPig.copy(pig.group.position).sub(this.camera.position);
            const dist = toPig.length();
            if (dist > meleeRange) continue;

            const dir = toPig.clone().normalize();
            const dot = dir.dot(forward);
            if (dot < 0.45) continue;

            if (dist < bestDist) {
                bestDist = dist;
                bestPig = pig;
            }
        }

        if (!bestPig) return;

        bestPig.takeDamage(3);
        this.showPickupFeedback(bestPig.hp > 0 ? `Pig Hit (${bestPig.hp}/10)` : 'Pig Down!');
    }

    consumeToolDurability(tool, amount = 1) {
        if (!tool) return;
        tool.durability -= amount;
        if (tool.durability <= 0) {
            this.removeToolById(tool.id);
            this.showPickupFeedback(tool.type === 'Stone Club' ? "Club Broke!" : "Axe Broke!");
        }
        this.updateInventoryUI();
    }

    handleAction() {
        if (this.placementMode) {
            this.placeObject();
            return;
        }
        const activeTool = this.getActiveTool();
        if (!activeTool) return;

        if (activeTool.type === 'Stone Axe') {
            this.swingTool(activeTool, this.axeModel, 1.5);
        } else if (activeTool.type === 'Stone Club') {
            this.swingTool(activeTool, this.clubModel, 1.35);
        }
    }

    swingTool(tool, model, swingAmount) {
        if (this.isSwinging || !model) return;
        this.isSwinging = true;

        // Visual swing
        const startRot = model.rotation.clone();
        const swingTime = 0.2;
        const startTime = performance.now();

        const animateSwing = () => {
            const now = performance.now();
            const elapsed = (now - startTime) / 1000;
            const t = Math.min(elapsed / swingTime, 1.0);

            if (t < 1.0) {
                model.rotation.x = startRot.x + Math.sin(t * Math.PI) * swingAmount;
                requestAnimationFrame(animateSwing);
            } else {
                model.rotation.copy(startRot);
                this.isSwinging = false;

                // Hit check
                this.resolveToolHit(tool);
            }
        };
        animateSwing();
    }

    resolveToolHit(tool) {
        if (tool?.type === 'Stone Club') {
            this.tryHitPig(tool);
        } else if (this.isChopTool(tool) && this.focusedItem && this.focusedItem.mesh.userData.isTree) {
            this.chopTree(this.focusedItem.mesh, this.focusedItem.index, tool);
        }

        this.consumeToolDurability(tool, 1);
    }

    chopTree(mesh, index, toolUsed) {
        let ownerChunk = null;
        for (const chunk of this.chunks.values()) {
            if (chunk.treeWoodMeshes.includes(mesh)) {
                ownerChunk = chunk;
                break;
            }
        }
        if (!ownerChunk) return;

        const matrix = new THREE.Matrix4();
        mesh.getMatrixAt(index, matrix);
        const pos = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        matrix.decompose(pos, quat, scale);

        if (scale.length() < 0.001) return;

        const emptyMatrix = new THREE.Matrix4().makeScale(0, 0, 0);
        mesh.setMatrixAt(index, emptyMatrix);
        mesh.instanceMatrix.needsUpdate = true;

        const leafMesh = mesh.userData.leafMesh;
        if (leafMesh) {
            leafMesh.setMatrixAt(index, emptyMatrix);
            leafMesh.instanceMatrix.needsUpdate = true;
        }

        const protoIdx = mesh.userData.protoIdx;
        const woodProto = this.treePrototypes[protoIdx].wood;
        const leafProto = this.treePrototypes[protoIdx].leaves;

        const fallingGroup = new THREE.Group();
        fallingGroup.position.copy(pos);
        fallingGroup.quaternion.copy(quat);
        fallingGroup.scale.copy(scale);

        // Simple non-sway materials for the falling model
        const fallWoodMat = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 0.9, flatShading: true });
        const fallLeafMat = new THREE.MeshStandardMaterial({ color: 0x2d5a27, roughness: 0.8, side: THREE.DoubleSide, flatShading: true });

        const wood = new THREE.Mesh(woodProto, fallWoodMat);
        const leaves = new THREE.Mesh(leafProto, fallLeafMat);

        fallingGroup.add(wood, leaves);
        this.scene.add(fallingGroup);

        const dx = Math.random() - 0.5;
        const dz = Math.random() - 0.5;
        const dir = new THREE.Vector3(dx === 0 ? 1 : dx, 0, dz === 0 ? 1 : dz).normalize();
        const axis = new THREE.Vector3(0, 1, 0).cross(dir).normalize();

        this.activeFallingTrees.push({
            group: fallingGroup,
            pos: pos.clone(),
            chunk: ownerChunk,
            vel: 0.002, // Slower start
            angle: 0,
            axis: axis,
            quat: quat.clone()
        });

    }

    spawnLogsAt(pos, chunk) {
        const count = 5 + Math.floor(Math.random() * 6);
        const logData = [];
        for (let i = 0; i < count; i++) {
            // Increase spread and add more variation
            const angle = Math.random() * Math.PI * 2;
            const dist = 4 + Math.random() * 12;
            logData.push({
                x: pos.x - chunk.x + Math.cos(angle) * dist,
                y: pos.y,
                z: pos.z - chunk.z + Math.sin(angle) * dist
            });
        }
        chunk.generateLogs(logData);
    }

    createSmashEffect(pos) {
        // Create an expanding shockwave ring
        const ringGeo = new THREE.TorusGeometry(0.5, 0.1, 8, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x998844, transparent: true, opacity: 0.8 });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.copy(pos);
        ring.position.y += 0.2; // Slightly above ground
        ring.rotation.x = Math.PI / 2;
        this.scene.add(ring);

        // Exploding bits
        const bitCount = 15;
        const bits = [];
        const bitGeo = new THREE.IcosahedronGeometry(0.4, 0);
        for (let i = 0; i < bitCount; i++) {
            const bit = new THREE.Mesh(bitGeo, new THREE.MeshStandardMaterial({ color: 0x887755 }));
            bit.position.copy(pos);
            bit.position.y += 1.0;
            const bitDir = new THREE.Vector3(
                (Math.random() - 0.5) * 1.5,
                Math.random() * 2.0,
                (Math.random() - 0.5) * 1.5
            );
            bits.push({ mesh: bit, vel: bitDir });
            this.scene.add(bit);
        }

        const animStart = performance.now();
        const runSmash = () => {
            const elapsed = (performance.now() - animStart) / 1000;
            const t = Math.min(elapsed / 1.0, 1.0); // 1-second duration

            // Expand ring
            ring.scale.set(1 + t * 15, 1 + t * 15, 1);
            ring.material.opacity = 0.8 * (1 - t);

            // Fly bits
            bits.forEach(b => {
                b.mesh.position.add(b.vel.clone().multiplyScalar(0.15));
                b.vel.y -= 0.05; // gravity
                b.mesh.rotation.x += 0.1;
                b.mesh.scale.multiplyScalar(0.98);
            });

            if (t < 1.0) {
                requestAnimationFrame(runSmash);
            } else {
                this.scene.remove(ring);
                bits.forEach(b => this.scene.remove(b.mesh));
            }
        };
        runSmash();
    }

    showPickupFeedback(name) {
        let feedback = document.getElementById('pickup-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.id = 'pickup-feedback';
            feedback.style.position = 'fixed';
            feedback.style.bottom = '100px';
            feedback.style.left = '50%';
            feedback.style.transform = 'translateX(-50%)';
            feedback.style.padding = '10px 20px';
            feedback.style.background = 'rgba(0,0,0,0.7)';
            feedback.style.color = 'white';
            feedback.style.borderRadius = '5px';
            feedback.style.pointerEvents = 'none';
            feedback.style.transition = 'opacity 0.3s';
            document.body.appendChild(feedback);
        }
        feedback.textContent = `Picked up: ${name}`;
        feedback.style.opacity = '1';
        clearTimeout(this.feedbackTimeout);
        this.feedbackTimeout = setTimeout(() => {
            feedback.style.opacity = '0';
        }, 2000);
    }

    createViewmodel() {
        this.viewmodelGroup = new THREE.Group();
        this.camera.add(this.viewmodelGroup); // Attach to camera

        // Stone Axe Model
        const axe = new THREE.Group();

        // Handle
        const handleGeo = new THREE.CylinderGeometry(0.05, 0.05, 1.2, 8);
        const handleMat = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 0.9 });
        const handle = new THREE.Mesh(handleGeo, handleMat);
        handle.rotation.z = Math.PI * 0.1;
        axe.add(handle);

        // Head - sharp wedge, oriented so the blade faces away from the player
        const headShape = new THREE.Shape();
        headShape.moveTo(-0.14, 0.12);
        headShape.lineTo(0.08, 0.14);
        headShape.lineTo(0.34, 0.0); // acute sharp edge
        headShape.lineTo(0.08, -0.14);
        headShape.lineTo(-0.14, -0.12);
        headShape.lineTo(-0.14, 0.12);

        const extrudeSettings = { depth: 0.12, bevelEnabled: true, bevelThickness: 0.006, bevelSize: 0.008, bevelSegments: 2 };
        const headGeo = new THREE.ExtrudeGeometry(headShape, extrudeSettings);
        headGeo.translate(0, 0, -0.06); // Center extrusion

        const headMat = new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.6, flatShading: true });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.set(0.1, 0.45, 0);
        head.rotation.y = Math.PI; // flip to point blade away from camera
        head.rotation.z = -Math.PI * 0.08;
        axe.add(head);

        // Positioning in front of camera
        axe.position.set(0.4, -0.5, -0.8);
        axe.rotation.set(0, -Math.PI * 0.5, 0.2);
        axe.userData.idlePos = axe.position.clone();

        this.axeModel = axe;
        this.viewmodelGroup.add(axe);

        // Stone Club Model
        const club = new THREE.Group();

        const clubHandleGeo = new THREE.CylinderGeometry(0.055, 0.07, 1.2, 8);
        const clubHandleMat = new THREE.MeshStandardMaterial({ color: 0x6d4c41, roughness: 0.92 });
        const clubHandle = new THREE.Mesh(clubHandleGeo, clubHandleMat);
        clubHandle.rotation.z = Math.PI * 0.08;
        club.add(clubHandle);

        const clubHeadMat = new THREE.MeshStandardMaterial({ color: 0x8a8a8a, roughness: 0.75, flatShading: true });
        const clubHeadCore = new THREE.Mesh(new THREE.SphereGeometry(0.18, 10, 8), clubHeadMat);
        clubHeadCore.position.set(0.08, 0.48, 0);
        club.add(clubHeadCore);

        const clubBump1 = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 6), clubHeadMat);
        clubBump1.position.set(0.18, 0.52, 0.03);
        club.add(clubBump1);

        const clubBump2 = new THREE.Mesh(new THREE.SphereGeometry(0.085, 8, 6), clubHeadMat);
        clubBump2.position.set(0.03, 0.56, -0.08);
        club.add(clubBump2);

        const clubBump3 = new THREE.Mesh(new THREE.SphereGeometry(0.075, 8, 6), clubHeadMat);
        clubBump3.position.set(0.02, 0.44, 0.09);
        club.add(clubBump3);

        club.position.set(0.4, -0.5, -0.8);
        club.rotation.set(0, -Math.PI * 0.5, 0.2);
        club.userData.idlePos = club.position.clone();

        this.clubModel = club;
        this.viewmodelGroup.add(club);
        this.updateViewmodel();
    }

    updateViewmodel() {
        if (!this.axeModel || !this.clubModel) return;
        const activeTool = this.getActiveTool();
        this.axeModel.visible = !!activeTool && activeTool.type === 'Stone Axe';
        this.clubModel.visible = !!activeTool && activeTool.type === 'Stone Club';
    }

    toggleCrafting() {
        this.craftingOpen = !this.craftingOpen;
        const panel = document.getElementById('crafting-panel');
        if (panel) panel.style.display = this.craftingOpen ? 'block' : 'none';

        if (this.craftingOpen && this.controls.locked) {
            document.exitPointerLock();
        }
    }

    craftStoneAxe() {
        if (this.inventory.branches >= 1 && this.inventory.stones >= 2) {
            this.inventory.branches -= 1;
            this.inventory.stones -= 2;
            this.addToolToInventory('Stone Axe', 100);
            this.updateInventoryUI();
            this.showPickupFeedback("Stone Axe");
        } else {
            this.showPickupFeedback("Missing Materials!");
        }
    }

    craftStoneClub() {
        if (this.inventory.logs >= 1 && this.inventory.stones >= 1) {
            this.inventory.logs -= 1;
            this.inventory.stones -= 1;
            this.addToolToInventory('Stone Club', 100);
            this.updateInventoryUI();
            this.showPickupFeedback("Stone Club");
        } else {
            this.showPickupFeedback("Missing Materials!");
        }
    }

    craftFirepit() {
        if (this.inventory.logs >= 5 && this.inventory.stones >= 2) {
            this.inventory.logs -= 5;
            this.inventory.stones -= 2;
            this.updateInventoryUI();
            this.startPlacement('firepit');
            this.toggleCrafting();
            this.showPickupFeedback("Placing Firepit...");
        } else {
            this.showPickupFeedback("Missing Materials!");
        }
    }

    setupCraftingListeners() {
        const axeBtn = document.getElementById('btn-craft-axe');
        if (axeBtn) axeBtn.onclick = () => this.craftStoneAxe();

        const clubBtn = document.getElementById('btn-craft-club');
        if (clubBtn) clubBtn.onclick = () => this.craftStoneClub();

        const firepitBtn = document.getElementById('btn-craft-firepit');
        if (firepitBtn) firepitBtn.onclick = () => this.craftFirepit();

        const wallBtn = document.getElementById('btn-craft-wall');
        if (wallBtn) wallBtn.onclick = () => this.craftWall();

        const floorBtn = document.getElementById('btn-craft-floor');
        if (floorBtn) floorBtn.onclick = () => this.craftFloor();

        const roofBtn = document.getElementById('btn-craft-roof');
        if (roofBtn) roofBtn.onclick = () => this.craftRoof();

        const triWallBtn = document.getElementById('btn-craft-tri-wall');
        if (triWallBtn) triWallBtn.onclick = () => this.craftTriWall();
    }

    craftTriWall() {
        if (this.inventory.logs >= 1) {
            this.inventory.logs -= 1;
            this.updateInventoryUI();
            this.startPlacement('tri_wall');
            this.toggleCrafting();
            this.showPickupFeedback("Placing Triangle Wall... (R to rotate)");
        } else {
            this.showPickupFeedback("Missing Materials!");
        }
    }

    craftRoof() {
        if (this.inventory.logs >= 2) {
            this.inventory.logs -= 2;
            this.updateInventoryUI();
            this.startPlacement('roof');
            this.toggleCrafting();
            this.showPickupFeedback("Placing Roof... (R to rotate)");
        } else {
            this.showPickupFeedback("Missing Materials!");
        }
    }

    craftFloor() {
        if (this.inventory.logs >= 2) {
            this.inventory.logs -= 2;
            this.updateInventoryUI();
            this.startPlacement('floor');
            this.toggleCrafting();
            this.showPickupFeedback("Placing Floor... (R to rotate)");
        } else {
            this.showPickupFeedback("Missing Materials!");
        }
    }

    craftWall() {
        if (this.inventory.logs >= 2) {
            this.inventory.logs -= 2;
            this.updateInventoryUI();
            this.startPlacement('wall');
            this.toggleCrafting();
            this.showPickupFeedback("Placing Wall... (R to rotate)");
        } else {
            this.showPickupFeedback("Missing Materials!");
        }
    }

    startPlacement(type) {
        if (this.placementGhost) {
            this.scene.remove(this.placementGhost);
            this.placementGhost = null;
        }
        this.placementMode = type;
        this.placementRotation = 0;
        this.placementFlipped = false;
        if (type === 'firepit') {
            const ghostGroup = new THREE.Group();
            const logGeo = new THREE.CylinderGeometry(0.3, 0.3, 2.4, 8);
            const logMat = new THREE.MeshBasicMaterial({ color: 0x3d2b1f, transparent: true, opacity: 0.5 });
            for (let i = 0; i < 5; i++) {
                const log = new THREE.Mesh(logGeo, logMat);
                log.rotation.z = Math.PI / 2;
                log.rotation.y = (i / 5) * Math.PI * 2;
                log.position.y = 0.3;
                ghostGroup.add(log);
            }
            this.placementGhost = ghostGroup;
            this.scene.add(ghostGroup);
        } else if (type === 'wall') {
            const ghostGroup = new THREE.Group();
            const width = 5.0;
            const logGeo = new THREE.CylinderGeometry(0.25, 0.25, width, 8);
            const logMat = new THREE.MeshBasicMaterial({ color: 0x5d4037, transparent: true, opacity: 0.5 });
            for (let i = 0; i < 10; i++) {
                const log = new THREE.Mesh(logGeo, logMat);
                log.rotation.z = Math.PI / 2;
                log.position.y = 0.25 + i * 0.5;
                ghostGroup.add(log);
            }
            this.placementGhost = ghostGroup;
            this.scene.add(ghostGroup);
        } else if (type === 'floor') {
            const ghostGroup = new THREE.Group();
            const size = 5.0;
            const plankGeo = new THREE.BoxGeometry(0.48, 0.1, size);
            const plankMat = new THREE.MeshBasicMaterial({ color: 0x795548, transparent: true, opacity: 0.5 });
            for (let i = 0; i < 10; i++) {
                const plank = new THREE.Mesh(plankGeo, plankMat);
                plank.position.x = -size / 2 + 0.25 + i * 0.5;
                ghostGroup.add(plank);
            }
            this.placementGhost = ghostGroup;
            this.scene.add(ghostGroup);
        } else if (type === 'roof') {
            const ghostGroup = new THREE.Group();
            const angle = Math.PI / 4;
            const width = 5.0;
            const length = 5.0 * Math.sqrt(2);
            const slopeGroup = new THREE.Group();
            slopeGroup.rotation.x = angle; // Sloping UPWARD
            ghostGroup.add(slopeGroup);
            const plankGeo = new THREE.BoxGeometry(0.48, 0.1, length);
            const plankMat = new THREE.MeshBasicMaterial({ color: 0x4e342e, transparent: true, opacity: 0.5 });
            for (let i = 0; i < 10; i++) {
                const plank = new THREE.Mesh(plankGeo, plankMat);
                plank.position.x = -width / 2 + 0.25 + i * 0.5;
                plank.position.z = -length / 2;
                slopeGroup.add(plank);
            }
            // Thatch preview
            const thatchGeo = new THREE.BoxGeometry(width, 0.15, length);
            const thatchMat = new THREE.MeshBasicMaterial({ color: 0xd4c28d, transparent: true, opacity: 0.5 });
            const thatch = new THREE.Mesh(thatchGeo, thatchMat);
            thatch.position.y = 0.1;
            thatch.position.z = -length / 2;
            slopeGroup.add(thatch);

            this.placementGhost = ghostGroup;
            this.scene.add(ghostGroup);
        } else if (type === 'tri_wall') {
            const ghostGroup = new THREE.Group();
            const width = 5.0;
            const logMat = new THREE.MeshBasicMaterial({ color: 0x5d4037, transparent: true, opacity: 0.5 });
            for (let i = 0; i < 10; i++) {
                const currentWidth = width * (1 - (i / 10));
                const logGeo = new THREE.CylinderGeometry(0.25, 0.25, currentWidth, 8);
                const log = new THREE.Mesh(logGeo, logMat);
                log.rotation.z = Math.PI / 2;
                log.position.y = 0.25 + i * 0.5;
                log.position.x = -currentWidth / 2;
                ghostGroup.add(log);
            }
            this.placementGhost = ghostGroup;
            this.scene.add(ghostGroup);
        }
    }

    placeObject() {
        if (this.placementMode === 'firepit' && this.placementGhost) {
            const firepit = new Firepit(this.scene, this.placementGhost.position);
            this.firepits.push(firepit);
            this.scene.remove(this.placementGhost);
            this.placementGhost = null;
            this.placementMode = null;
            this.showPickupFeedback("Firepit Placed!");
        } else if (this.placementMode === 'wall' && this.placementGhost) {
            const wall = new Wall(this.scene, this.placementGhost.position, this.placementGhost.rotation.y);
            this.walls.push(wall);
            this.scene.remove(this.placementGhost);
            this.placementGhost = null;
            this.placementMode = null;
            this.showPickupFeedback("Wall Placed!");
        } else if (this.placementMode === 'floor' && this.placementGhost) {
            const floor = new Floor(this.scene, this.placementGhost.position, this.placementGhost.rotation.y);
            this.floors.push(floor);
            this.scene.remove(this.placementGhost);
            this.placementGhost = null;
            this.placementMode = null;
            this.showPickupFeedback("Floor Placed!");
        } else if (this.placementMode === 'roof' && this.placementGhost) {
            const roof = new Roof(this.scene, this.placementGhost.position, this.placementGhost.rotation.y, this.thatchMaterial);
            this.roofs.push(roof);
            this.scene.remove(this.placementGhost);
            this.placementGhost = null;
            this.placementMode = null;
            this.showPickupFeedback("Roof Placed!");
        } else if (this.placementMode === 'tri_wall' && this.placementGhost) {
            const triWall = new WallTriangle(this.scene, this.placementGhost.position, this.placementGhost.rotation.y, this.placementFlipped);
            this.triWalls.push(triWall);
            this.scene.remove(this.placementGhost);
            this.placementGhost = null;
            this.placementMode = null;
            this.showPickupFeedback("Triangle Wall Placed!");
        }
    }

    updateInventoryUI() {
        let invDiv = document.getElementById('inventory-panel');
        if (!invDiv) {
            invDiv = document.createElement('div');
            invDiv.id = 'inventory-panel';
            invDiv.style.marginTop = '15px';
            invDiv.style.paddingTop = '10px';
            invDiv.style.borderTop = '1px solid rgba(255,255,255,0.2)';
            document.getElementById('info-panel').appendChild(invDiv);
        }
        invDiv.innerHTML = `<strong>Inventory:</strong><br>
            Branches: ${this.inventory.branches}<br>
            Berries: ${this.inventory.berries}<br>
            Stones: ${this.inventory.stones}<br>
            Logs: ${this.inventory.logs}<br>
            Meat: ${this.inventory.meat}<br>
            Leather: ${this.inventory.leather}<br>
            Red Mushrooms: ${this.inventory.redMushrooms}<br>
            Yellow Mushrooms: ${this.inventory.yellowMushrooms}`;

        if (this.inventory.tools.length > 0) {
            invDiv.innerHTML += `<br><strong>Tools:</strong>`;
            this.inventory.tools.forEach(tool => {
                invDiv.innerHTML += `<br>${tool.type} (${tool.durability}%)`;
            });
        }

        this.updateViewmodel();
        this.updateActionbarUI();
    }

    updateInteraction() {
        if (!this.controls.locked) {
            this.highlightMesh.visible = false;
            this.focusedItem = null;
            return;
        }

        this.raycaster.setFromCamera({ x: 0, y: 0 }, this.camera);

        // Get all interactable meshes
        const interactables = [];
        const activeTool = this.getActiveTool();
        const canChopTrees = this.isChopTool(activeTool);

        this.chunks.forEach(chunk => {
            interactables.push(...chunk.branchMeshes);
            interactables.push(...chunk.bushMeshes);
            interactables.push(...chunk.stoneMeshes);
            interactables.push(...chunk.logMeshes);
            interactables.push(...chunk.mushroomMeshes);
            if (canChopTrees) {
                interactables.push(...chunk.treeWoodMeshes);
            }
        });
        interactables.push(...this.droppedItems);

        const intersects = this.raycaster.intersectObjects(interactables);

        // Placement Mode Logic
        if (this.placementMode && this.placementGhost) {
            // Update rotation
            this.placementGhost.rotation.y = (this.placementRotation / 16) * Math.PI * 2;

            // Find ground or wall or floor or roof intersection
            const buildables = [
                ...this.walls.map(w => w.group),
                ...this.triWalls.map(tw => tw.group),
                ...this.floors.map(f => f.group),
                ...this.roofs.map(r => r.group)
            ];
            const groundIntersects = this.raycaster.intersectObjects([this.terrainGroup, ...buildables], true);
            if (groundIntersects.length > 0) {
                const hit = groundIntersects[0];
                this.placementGhost.position.copy(hit.point);

                // Snapping Logic
                if (this.placementMode === 'wall' || this.placementMode === 'tri_wall' || this.placementMode === 'floor' || this.placementMode === 'roof') {
                    const snapDistance = 3.0;
                    const ghostWidth = 5.0;

                    let bestSnapPoint = null;
                    let minDist = snapDistance;

                    // Check all snap points from walls, floors, and roofs
                    const allSnapPoints = [];
                    this.walls.forEach(w => allSnapPoints.push(...w.getSnapPoints()));
                    this.triWalls.forEach(tw => allSnapPoints.push(...tw.getSnapPoints()));
                    this.floors.forEach(f => allSnapPoints.push(...f.getSnapPoints()));
                    this.roofs.forEach(r => allSnapPoints.push(...r.getSnapPoints()));

                    allSnapPoints.forEach(sp => {
                        const d = sp.pos.distanceTo(hit.point);
                        if (d < minDist) {
                            minDist = d;
                            bestSnapPoint = sp;
                        }
                    });

                    if (bestSnapPoint) {
                        if (this.placementMode === 'wall') {
                            if (bestSnapPoint.type.includes('top')) {
                                this.placementGhost.position.copy(bestSnapPoint.pos);
                                this.placementGhost.rotation.y = bestSnapPoint.rotation;
                            } else if (bestSnapPoint.type.includes('bottom') || bestSnapPoint.type.includes('floor_edge') || bestSnapPoint.type.includes('roof_bottom')) {
                                // Snap wall to side/corner of another wall or edge of a floor
                                const leftOffset = new THREE.Vector3(-ghostWidth / 2, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.placementGhost.rotation.y);
                                const rightOffset = new THREE.Vector3(ghostWidth / 2, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.placementGhost.rotation.y);
                                const posIfLeft = bestSnapPoint.pos.clone().sub(leftOffset);
                                const posIfRight = bestSnapPoint.pos.clone().sub(rightOffset);
                                this.placementGhost.position.copy(posIfLeft.distanceTo(hit.point) < posIfRight.distanceTo(hit.point) ? posIfLeft : posIfRight);
                            }
                        } else if (this.placementMode === 'floor') {
                            const offsets = [
                                new THREE.Vector3(-2.5, 0, 0), new THREE.Vector3(2.5, 0, 0),
                                new THREE.Vector3(0, 0, -2.5), new THREE.Vector3(0, 0, 2.5)
                            ];
                            let bestFloorPos = null;
                            let innerMin = 1000;

                            offsets.forEach(off => {
                                const rotatedOff = off.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), this.placementGhost.rotation.y);
                                const potentialPos = bestSnapPoint.pos.clone().sub(rotatedOff);
                                const d = potentialPos.distanceTo(hit.point);
                                if (d < innerMin) {
                                    innerMin = d;
                                    bestFloorPos = potentialPos;
                                }
                            });
                            this.placementGhost.position.copy(bestFloorPos);
                        } else if (this.placementMode === 'roof') {
                            if (bestSnapPoint.type === 'roof_side') {
                                // Modular Roof Snapping: Align side to side
                                const angle = Math.PI / 4; // Roof angle
                                const width = 5.0;
                                const length = width * Math.sqrt(2);
                                const sideOffset_X = width / 2;
                                const sideOffset_Slope = length / 2;

                                // Local relative position of our side points
                                const leftRel = new THREE.Vector3(-sideOffset_X, sideOffset_Slope * Math.sin(angle), -sideOffset_Slope * Math.cos(angle)).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.placementGhost.rotation.y);
                                const rightRel = new THREE.Vector3(sideOffset_X, sideOffset_Slope * Math.sin(angle), -sideOffset_Slope * Math.cos(angle)).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.placementGhost.rotation.y);

                                const posIfLeftSnaps = bestSnapPoint.pos.clone().sub(leftRel);
                                const posIfRightSnaps = bestSnapPoint.pos.clone().sub(rightRel);

                                if (posIfLeftSnaps.distanceTo(hit.point) < posIfRightSnaps.distanceTo(hit.point)) {
                                    this.placementGhost.position.copy(posIfLeftSnaps);
                                } else {
                                    this.placementGhost.position.copy(posIfRightSnaps);
                                }
                                // Match rotation for modular tiling
                                this.placementGhost.rotation.y = bestSnapPoint.rotation;
                            } else {
                                // Default: Align bottom of ghost to target snap point
                                this.placementGhost.position.copy(bestSnapPoint.pos);
                            }
                        } else if (this.placementMode === 'tri_wall') {
                            // Triangle Wall Snapping
                            // Goal: Match triangle corners to building corners
                            this.placementGhost.scale.x = this.placementFlipped ? -1 : 1;

                            if (bestSnapPoint.type.includes('top') || bestSnapPoint.type.includes('corner')) {
                                const flipFactor = this.placementFlipped ? -1 : 1;
                                const corner90Rel = new THREE.Vector3(0, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.placementGhost.rotation.y);
                                const cornerThinRel = new THREE.Vector3(-5 * flipFactor, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.placementGhost.rotation.y);

                                const posIf90Snaps = bestSnapPoint.pos.clone().sub(corner90Rel);
                                const posIfThinSnaps = bestSnapPoint.pos.clone().sub(cornerThinRel);

                                if (posIf90Snaps.distanceTo(hit.point) < posIfThinSnaps.distanceTo(hit.point)) {
                                    this.placementGhost.position.copy(posIf90Snaps);
                                } else {
                                    this.placementGhost.position.copy(posIfThinSnaps);
                                }

                                // Match rotation for modular tiling
                                this.placementGhost.rotation.y = bestSnapPoint.rotation;
                            } else {
                                // Default fallback
                                this.placementGhost.position.copy(bestSnapPoint.pos);
                            }
                        }
                    }
                }

                if (this.placementMode === 'firepit') this.placementGhost.position.y += 0.05;
                this.placementGhost.visible = true;
            } else {
                this.placementGhost.visible = false;
            }
            this.highlightMesh.visible = false;
            return;
        }

        if (intersects.length > 0) {
            const hit = intersects[0];
            const mesh = hit.object;
            const index = hit.instanceId;

            if (index !== undefined && typeof mesh.getMatrixAt === 'function') {
                // Matrix extraction for InstancedMesh
                const matrix = new THREE.Matrix4();
                mesh.getMatrixAt(index, matrix);
                const scale = new THREE.Vector3();
                const pos = new THREE.Vector3();
                const quat = new THREE.Quaternion();
                matrix.decompose(pos, quat, scale);

                if (scale.length() > 0.001) {
                    // For bushes, check if berries are still there
                    if (mesh.userData.isBush) {
                        const berryMesh = mesh.userData.berryMesh;
                        const bMatrix = new THREE.Matrix4();
                        berryMesh.getMatrixAt(index, bMatrix);
                        const bScale = new THREE.Vector3();
                        bMatrix.decompose(new THREE.Vector3(), new THREE.Quaternion(), bScale);
                        if (bScale.length() < 0.001) {
                            this.highlightMesh.visible = false;
                            this.focusedItem = null;
                            return;
                        }
                    }

                    this.focusedItem = { mesh, index };

                    // Update highlight mesh to match the instance
                    this.highlightMesh.geometry = mesh.geometry;
                    this.highlightMesh.position.copy(pos);
                    this.highlightMesh.quaternion.copy(quat);
                    this.highlightMesh.scale.copy(scale).multiplyScalar(1.05);
                    this.highlightMesh.visible = true;
                    return;
                }
            } else if (mesh.userData.isDrop) {
                this.focusedItem = { mesh, index: null };

                const pos = new THREE.Vector3();
                const quat = new THREE.Quaternion();
                const scale = new THREE.Vector3();
                mesh.getWorldPosition(pos);
                mesh.getWorldQuaternion(quat);
                mesh.getWorldScale(scale);

                this.highlightMesh.geometry = mesh.geometry;
                this.highlightMesh.position.copy(pos);
                this.highlightMesh.quaternion.copy(quat);
                this.highlightMesh.scale.copy(scale).multiplyScalar(1.2);
                this.highlightMesh.visible = true;
                return;
            }
        }

        this.focusedItem = null;
        this.highlightMesh.visible = false;
    }

    setupUI() {
        // Crosshair
        if (!document.getElementById('crosshair')) {
            const crosshair = document.createElement('div');
            crosshair.id = 'crosshair';
            crosshair.style.position = 'fixed';
            crosshair.style.top = '50%';
            crosshair.style.left = '50%';
            crosshair.style.width = '12px';
            crosshair.style.height = '12px';
            crosshair.style.marginLeft = '-6px';
            crosshair.style.marginTop = '-6px';
            crosshair.style.border = '2px solid rgba(255, 255, 255, 0.8)';
            crosshair.style.borderRadius = '50%';
            crosshair.style.pointerEvents = 'none';
            crosshair.style.zIndex = '1000';
            crosshair.style.boxShadow = '0 0 4px rgba(0,0,0,0.5)';
            document.body.appendChild(crosshair);

            // Center dot
            const dot = document.createElement('div');
            dot.style.position = 'absolute';
            dot.style.top = '50%';
            dot.style.left = '50%';
            dot.style.width = '2px';
            dot.style.height = '2px';
            dot.style.marginLeft = '-1px';
            dot.style.marginTop = '-1px';
            dot.style.background = 'white';
            dot.style.borderRadius = '50%';
            crosshair.appendChild(dot);
        }

        const speedSlider = document.getElementById('speed-slider');
        const speedDisplay = document.getElementById('speed-display');

        const updateSpeedDisplay = () => {
            speedDisplay.textContent = CONFIG.camera.moveSpeed < 10 ? CONFIG.camera.moveSpeed.toFixed(1) : Math.round(CONFIG.camera.moveSpeed);
        };
        updateSpeedDisplay();

        if (speedSlider) {
            speedSlider.addEventListener('input', (e) => {
                CONFIG.camera.moveSpeed = parseFloat(e.target.value);
                updateSpeedDisplay();
            });
            speedSlider.addEventListener('mousedown', (e) => e.stopPropagation());
        }

        this.fpsDisplay = document.getElementById('fps-display');
        this.trisDisplay = document.getElementById('tris-display');

        const uiContainer = document.getElementById('info-panel');

        // Crafting Button Binding
        const craftBtn = document.getElementById('btn-craft-axe');
        if (craftBtn) {
            craftBtn.addEventListener('mousedown', (e) => e.stopPropagation());
        }

        const craftClubBtn = document.getElementById('btn-craft-club');
        if (craftClubBtn) {
            craftClubBtn.addEventListener('mousedown', (e) => e.stopPropagation());
        }

        // Rain Toggle
        if (!document.getElementById('btn-rain')) {
            const div = document.createElement('div');
            div.style.marginTop = '10px';
            div.innerHTML = `<button id="btn-rain" style="width:100%; padding: 5px; cursor: pointer;">Toggle Rain</button>`;
            uiContainer.appendChild(div);

            document.getElementById('btn-rain').addEventListener('click', (e) => {
                e.stopPropagation();
                if (this.rainEffect.enabled) {
                    this.rainEffect.disable();
                    e.target.style.background = '';
                    e.target.style.color = '';
                } else {
                    this.rainEffect.enable();
                    e.target.style.background = '#4a90e2';
                    e.target.style.color = 'white';
                }
            });
            document.getElementById('btn-rain').addEventListener('mousedown', (e) => e.stopPropagation());
        }

        // Mist Toggle
        if (!document.getElementById('btn-mist')) {
            const div = document.createElement('div');
            div.style.marginTop = '10px';
            div.innerHTML = `<button id="btn-mist" style="width:100%; padding: 5px; cursor: pointer;">Toggle Mist</button>`;
            uiContainer.appendChild(div);

            document.getElementById('btn-mist').addEventListener('click', (e) => {
                e.stopPropagation();
                this.mistEnabled = !this.mistEnabled;

                if (this.mistEnabled) {
                    // Enter Mist - very close view distance
                    this.scene.background.set(0xcccccc);
                    this.scene.fog.color.set(0xcccccc);
                    this.scene.fog.near = 0;
                    this.scene.fog.far = 300;
                    e.target.textContent = "Disable Mist";
                    e.target.style.background = "#999999";
                    e.target.style.color = "white";
                } else {
                    // Exit Mist - restore original blue sky and view distance
                    this.scene.background.set(0x87ceeb);
                    this.scene.fog.color.set(0x87ceeb);
                    this.scene.fog.near = this.viewDistance * 0.5;
                    this.scene.fog.far = this.viewDistance;
                    e.target.textContent = "Toggle Mist";
                    e.target.style.background = "";
                    e.target.style.color = "";
                }
            });
            document.getElementById('btn-mist').addEventListener('mousedown', (e) => e.stopPropagation());
        }

        // Wind Toggle
        if (!document.getElementById('btn-wind')) {
            const div = document.createElement('div');
            div.style.marginTop = '10px';
            div.innerHTML = `<button id="btn-wind" style="width:100%; padding: 5px; cursor: pointer;">Toggle Wind</button>`;
            uiContainer.appendChild(div);

            document.getElementById('btn-wind').addEventListener('click', (e) => {
                e.stopPropagation();
                this.windEnabled = !this.windEnabled;
                if (this.windEnabled) {
                    e.target.style.background = '#4caf50';
                    e.target.style.color = 'white';
                    this.windUniforms.uWindStrength.value = 5.0; // High strength for effect
                } else {
                    e.target.style.background = '';
                    e.target.style.color = '';
                    this.windUniforms.uWindStrength.value = 0.0;
                }
            });
            document.getElementById('btn-wind').addEventListener('mousedown', (e) => e.stopPropagation());
        }

        // Volumetric Clouds Toggle
        if (!document.getElementById('btn-clouds')) {
            const div = document.createElement('div');
            div.style.marginTop = '10px';
            div.innerHTML = `<button id="btn-clouds" style="width:100%; padding: 5px; cursor: pointer; background: #f5f5f5;">Disable Clouds</button>`;
            uiContainer.appendChild(div);

            document.getElementById('btn-clouds').addEventListener('click', (e) => {
                e.stopPropagation();
                this.cloudsEnabled = !this.cloudsEnabled;
                if (this.cloudSystem) {
                    this.cloudSystem.setEnabled(this.cloudsEnabled);
                }
                if (this.cloudsEnabled) {
                    e.target.textContent = "Disable Clouds";
                    e.target.style.background = "#f5f5f5";
                    e.target.style.color = "";
                } else {
                    e.target.textContent = "Enable Clouds";
                    e.target.style.background = "";
                    e.target.style.color = "";
                }
            });
            document.getElementById('btn-clouds').addEventListener('mousedown', (e) => e.stopPropagation());
        }

        // View Distance Slider
        // Check if already exists to avoid dupes on reload
        if (!document.getElementById('view-dist-slider')) {
            const p = document.createElement('p');
            p.style.marginTop = '15px';
            p.innerHTML = `<strong>View Distance:</strong> <span id="view-dist-val">${this.viewDistance}</span><br>
            <input type="range" id="view-dist-slider" min="2000" max="25000" step="1000" value="${this.viewDistance}" style="width:100%">`;
            uiContainer.insertBefore(p, uiContainer.lastElementChild);

            const slider = document.getElementById('view-dist-slider');
            const valDisplay = document.getElementById('view-dist-val');
            slider.addEventListener('input', (e) => {
                this.viewDistance = parseInt(e.target.value);
                valDisplay.textContent = this.viewDistance;
                this.scene.fog.near = this.viewDistance * 0.5;
                this.scene.fog.far = this.viewDistance;
            });
            // Stop propagation on slider to prevent camera lock
            slider.addEventListener('mousedown', (e) => e.stopPropagation());
            slider.addEventListener('click', (e) => e.stopPropagation());
        }
        // Mode Toggle
        if (!document.getElementById('btn-mode')) {
            const div = document.createElement('div');
            div.style.marginTop = '10px';
            div.innerHTML = `<button id="btn-mode" style="width:100%; padding: 5px; cursor: pointer; background: #ff9800; color: white; border: none; border-radius: 4px;">Mode: Free-Fly</button>`;
            uiContainer.appendChild(div);

            document.getElementById('btn-mode').addEventListener('click', (e) => {
                e.stopPropagation();
                if (this.controls.mode === 'freefly') {
                    this.controls.mode = 'walk';
                    e.target.textContent = "Mode: Walking";
                    e.target.style.background = "#4CAF50";
                } else {
                    this.controls.mode = 'freefly';
                    e.target.textContent = "Mode: Free-Fly";
                    e.target.style.background = "#ff9800";
                }
            });
            document.getElementById('btn-mode').addEventListener('mousedown', (e) => e.stopPropagation());
        }

        this.setupActionbar();
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const time = performance.now();
        const delta = 0.016;

        this.controls.update(delta);

        // Update Chunks
        if (!this.lastChunkUpdate || time - this.lastChunkUpdate > 500) {
            this.updateChunks();
            this.lastChunkUpdate = time;
        }

        if (this.waterMaterial) {
            this.waterMaterial.uniforms.time.value += delta;
        }

        const seconds = time * 0.001;
        if (this.rainEffect) {
            this.rainEffect.update(delta, seconds);
        }
        if (this.splashSystem) {
            this.splashSystem.update(seconds);
        }

        if (this.windEnabled) {
            this.windTime += delta;
            this.windUniforms.uWindTime.value = this.windTime;
        }

        if (this.cloudSystem) {
            this.cloudSystem.update(delta, this.camera.position);
        }

        this.updateDroppedItemsPhysics(delta);

        // Viewmodel Bobbing
        const activeTool = this.getActiveTool();
        const activeModel = activeTool?.type === 'Stone Axe'
            ? this.axeModel
            : (activeTool?.type === 'Stone Club' ? this.clubModel : null);
        if (activeModel && activeModel.visible) {
            const isMoving = this.controls.velocity.lengthSq() > 0.1;
            const bobSpeed = 10;
            const bobAmount = 0.05;
            const idlePos = activeModel.userData.idlePos || new THREE.Vector3(0.4, -0.5, -0.8);

            if (isMoving) {
                activeModel.position.y = idlePos.y + Math.sin(time * 0.01 * bobSpeed) * bobAmount;
                activeModel.position.x = idlePos.x + Math.cos(time * 0.01 * (bobSpeed * 0.5)) * (bobAmount * 0.5);
            } else {
                // Return to idle
                activeModel.position.y += (idlePos.y - activeModel.position.y) * 0.1;
                activeModel.position.x += (idlePos.x - activeModel.position.x) * 0.1;
            }
        }

        this.updateInteraction();

        // Update Animals
        for (let i = this.pigs.length - 1; i >= 0; i--) {
            this.pigs[i].update(delta, seconds, this.camera);
        }
        this.rabbits.forEach(rabbit => rabbit.update(delta, seconds));
        this.firepits.forEach(fp => fp.update(delta));

        // Update Falling Trees
        for (let i = this.activeFallingTrees.length - 1; i >= 0; i--) {
            const tree = this.activeFallingTrees[i];
            tree.vel += 0.0005; // Much slower acceleration
            tree.angle += tree.vel;

            tree.group.quaternion.copy(tree.quat);
            const tilt = new THREE.Quaternion().setFromAxisAngle(tree.axis, tree.angle);
            tree.group.quaternion.premultiply(tilt);

            if (tree.angle >= Math.PI * 0.48) {
                this.createSmashEffect(tree.pos);
                this.spawnLogsAt(tree.pos, tree.chunk);
                this.scene.remove(tree.group);
                this.activeFallingTrees.splice(i, 1);
            }
        }

        this.renderer.render(this.scene, this.camera);

        // Update Stats
        this.frameCount = (this.frameCount || 0) + 1;
        this.lastStatsUpdate = this.lastStatsUpdate || time;

        if (time >= this.lastStatsUpdate + 1000) {
            const fps = Math.round((this.frameCount * 1000) / (time - this.lastStatsUpdate));
            if (this.fpsDisplay) this.fpsDisplay.textContent = fps;
            if (this.trisDisplay && this.renderer.info) {
                this.trisDisplay.textContent = this.renderer.info.render.triangles.toLocaleString();
            }

            this.frameCount = 0;
            this.lastStatsUpdate = time;
        }
    }
}

// ===== START =====
// Dynamic import for BufferGeometryUtils
// We need to act carefully cause we are inside a module script but replacing content that might be after imports.
// The user's file has standard imports at top. We should restructure the file to add this import at top.

new TerrainScene();
