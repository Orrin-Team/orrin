use std::collections::VecDeque;

use sysinfo::{Pid, ProcessesToUpdate, System};

const HISTORY: usize = 240;

// Smaller = steadier readout, slower to react.
const EMA_ALPHA: f32 = 0.1;

// Sampling process memory every frame is wasteful and the number barely moves.
const MEM_REFRESH_SECS: f32 = 0.5;

pub struct FrameStats {
    frame_ms: VecDeque<f32>,
    avg_ms: f32,
    memory_bytes: u64,
    mem_accum: f32,
    sys: System,
    pid: Option<Pid>,
    gpu_ms: Option<f32>,
    gpu_ms_history: VecDeque<f32>,
    vram_used: Option<u64>,
    vram_total: u64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameStats {
    pub fn new() -> Self {
        Self {
            frame_ms: VecDeque::with_capacity(HISTORY),
            avg_ms: 0.0,
            memory_bytes: 0,
            // Sample memory on the very first frame.
            mem_accum: MEM_REFRESH_SECS,
            sys: System::new(),
            pid: sysinfo::get_current_pid().ok(),
            gpu_ms: None,
            gpu_ms_history: VecDeque::with_capacity(HISTORY),
            vram_used: None,
            vram_total: 0,
        }
    }

    pub fn record(&mut self, dt: f32) {
        let ms = dt * 1000.0;
        if self.frame_ms.len() == HISTORY {
            self.frame_ms.pop_front();
        }
        self.frame_ms.push_back(ms);

        self.avg_ms = if self.avg_ms == 0.0 {
            ms
        } else {
            self.avg_ms + EMA_ALPHA * (ms - self.avg_ms)
        };

        self.mem_accum += dt;
        if self.mem_accum >= MEM_REFRESH_SECS {
            self.mem_accum = 0.0;
            self.refresh_memory();
        }
    }

    fn refresh_memory(&mut self) {
        let Some(pid) = self.pid else {
            return;
        };
        self.sys
            .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        if let Some(proc) = self.sys.process(pid) {
            self.memory_bytes = proc.memory();
        }
    }

    pub fn set_gpu_stats(
        &mut self,
        gpu_ms: Option<f32>,
        vram_used: Option<u64>,
        vram_total: u64,
    ) {
        self.gpu_ms = gpu_ms;
        if let Some(ms) = gpu_ms {
            if self.gpu_ms_history.len() == HISTORY {
                self.gpu_ms_history.pop_front();
            }
            self.gpu_ms_history.push_back(ms);
        }
        self.vram_used = vram_used;
        self.vram_total = vram_total;
    }

    pub fn gpu_ms(&self) -> Option<f32> {
        self.gpu_ms
    }

    pub fn gpu_history(&self) -> &VecDeque<f32> {
        &self.gpu_ms_history
    }

    pub fn vram_used(&self) -> Option<u64> {
        self.vram_used
    }

    pub fn vram_total(&self) -> u64 {
        self.vram_total
    }

    pub fn fps(&self) -> f32 {
        if self.avg_ms > 0.0 {
            1000.0 / self.avg_ms
        } else {
            0.0
        }
    }

    pub fn frame_ms(&self) -> f32 {
        self.avg_ms
    }

    pub fn min_max_ms(&self) -> (f32, f32) {
        if self.frame_ms.is_empty() {
            return (0.0, 0.0);
        }
        self.frame_ms
            .iter()
            .copied()
            .fold((f32::MAX, 0.0), |(lo, hi), ms| (lo.min(ms), hi.max(ms)))
    }

    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub fn history(&self) -> &VecDeque<f32> {
        &self.frame_ms
    }
}
