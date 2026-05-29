import { create } from 'zustand';
import {
  EnergyState,
  TitanStatus,
  MoodStatus,
  WSEvent,
  MemoryNode,
} from '@/lib/types';

interface TitanStore {
  // Core state
  energyState: EnergyState;
  status: TitanStatus | null;
  mood: MoodStatus | null;
  wsConnected: boolean;

  // Event feed
  events: WSEvent[];
  maxEvents: number;

  // Selected memory node (for Neural tab)
  selectedNode: MemoryNode | null;

  // Maker state
  isMaker: boolean;
  makerPubkey: string | null;

  // Actions
  setEnergyState: (state: EnergyState) => void;
  setStatus: (status: TitanStatus) => void;
  setMood: (mood: MoodStatus) => void;
  setWSConnected: (connected: boolean) => void;
  pushEvent: (event: WSEvent) => void;
  setSelectedNode: (node: MemoryNode | null) => void;
  setMakerState: (isMaker: boolean, pubkey: string | null) => void;
}

export const useTitanStore = create<TitanStore>((set) => ({
  energyState: 'UNKNOWN',
  status: null,
  mood: null,
  wsConnected: false,
  events: [],
  maxEvents: 100,
  selectedNode: null,
  isMaker: false,
  makerPubkey: null,

  setEnergyState: (energyState) => set({ energyState }),

  setStatus: (status) =>
    set({
      status,
      energyState: status.energy_state,
    }),

  setMood: (mood) => set({ mood }),

  setWSConnected: (wsConnected) => set({ wsConnected }),

  pushEvent: (event) =>
    set((state) => ({
      events: [event, ...state.events].slice(0, state.maxEvents),
    })),

  setSelectedNode: (selectedNode) => set({ selectedNode }),

  setMakerState: (isMaker, makerPubkey) => set({ isMaker, makerPubkey }),
}));
