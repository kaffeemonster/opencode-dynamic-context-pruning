import * as fs from "fs/promises"
import { existsSync } from "fs"
import { gzip } from "zlib"
import { promisify } from "util"

const gzipAsync = promisify(gzip)

/**
 * Logrotate-style rotation for VCC view files.
 *
 * Before a new compile overwrites `{base}.txt` / `{base}.min.txt` /
 * `{base}.view.txt`, this shifts existing archives one slot up and
 * compresses the current file into `{base}.{suffix}.1.gz`. The newest
 * keep archives survive; older ones are removed.
 *
 * Layout after rotation (keep=3):
 *   current file     (fresh, about to be overwritten)
 *   .1.gz            most recent previous view
 *   .2.gz
 *   .3.gz            oldest retained
 */
export async function rotateViewFiles(
    exportPath: string,
    keep: number,
): Promise<string[]> {
    if (!keep || keep < 1) {
        return []
    }

    const rotated: string[] = []
    const base = exportPath.replace(/\.jsonl$/, "")
    const suffixes = [".txt", ".min.txt", ".view.txt"]

    for (const suffix of suffixes) {
        const current = `${base}${suffix}`
        if (!existsSync(current)) {
            continue
        }

        // Shift existing archives down: .N.gz -> .N+1.gz
        for (let i = keep - 1; i >= 1; i--) {
            const from = `${current}.${i}.gz`
            const to = `${current}.${i + 1}.gz`
            if (existsSync(from)) {
                try {
                    await fs.rename(from, to)
                } catch {}
            }
        }

        // Compress current file into .1.gz
        const dest = `${current}.1.gz`
        try {
            const content = await fs.readFile(current)
            const compressed = await gzipAsync(content)
            await fs.writeFile(dest, compressed)
            rotated.push(dest)
        } catch {}
    }

    return rotated
}
