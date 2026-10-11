import { type Plugin } from 'vite';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readStore, writeStore, checkLocal, excerpt, httpError, MAX_BYTES, FILES } from './server/store.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../..');
const content = path.join(here, 'content');

/** Dev-server API: read/write ONLY content/ideas.json and content/decisions.json; read-only evidence excerpts. */
function explorerApi(): Plugin {
  return {
    name: 'pravah-explorer-api',
    configureServer(server) {
      server.middlewares.use('/api', (req, res) => {
        const send = (status: number, body: unknown) => {
          res.statusCode = status;
          res.setHeader('content-type', 'application/json');
          res.setHeader('cache-control', 'no-store');
          res.end(JSON.stringify(body));
        };
        const fail = (e: any) => send(e.status ?? 500, { error: e.status ? e.message : 'server error' });
        try {
          const url = new URL(req.url ?? '/', 'http://x');
          const parts = url.pathname.split('/').filter(Boolean);
          const write = req.method === 'PUT';
          checkLocal(req, write);
          if (parts[0] === 'excerpt' && req.method === 'GET') {
            const model = JSON.parse(fs.readFileSync(path.join(content, 'model.json'), 'utf8'));
            return send(200, excerpt(repoRoot, model, url.searchParams.get('id') ?? '', 0));
          }
          const name = parts[0];
          if (parts.length !== 1 || !Object.prototype.hasOwnProperty.call(FILES, name)) throw httpError(404, 'not found');
          if (req.method === 'GET') return send(200, readStore(content, name));
          if (!write) throw httpError(405, 'method not allowed');
          const chunks: Buffer[] = [];
          let size = 0;
          req.on('data', (c: Buffer) => {
            size += c.length;
            if (size > MAX_BYTES) { req.destroy(); return; }
            chunks.push(c);
          });
          req.on('end', () => {
            try {
              const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
              send(200, writeStore(content, name, body.items, body.rev));
            } catch (e: any) { fail(e instanceof SyntaxError ? httpError(400, 'invalid JSON') : e); }
          });
        } catch (e) { fail(e); }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), explorerApi()],
  server: { host: '127.0.0.1', port: 5174, strictPort: true, fs: { allow: [here] } },
  test: { include: ['tests/**/*.test.ts'], environment: 'node' },
});
