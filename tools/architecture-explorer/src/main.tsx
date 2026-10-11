import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './styles.css';
import { App } from './App';
import model from '../content/model.json';
import type { Model } from './types';

createRoot(document.getElementById('root')!).render(<StrictMode><App model={model as unknown as Model} /></StrictMode>);
