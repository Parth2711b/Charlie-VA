const { makeWASocket, useMultiFileAuthState } = require('@whiskeysockets/baileys');
const pino = require('pino');
const path = require('path');
const fs = require('fs');
const qrcode = require('qrcode-terminal');

async function sendWhatsAppMessage(phone, message) {
    const authDir = path.join(__dirname, 'auth_info_baileys');
    const { state, saveCreds } = await useMultiFileAuthState(authDir);

    const sock = makeWASocket({
        auth: state,
        printQRInTerminal: true,
        logger: pino({ level: 'silent' }) // suppress annoying logs
    });

    sock.ev.on('creds.update', saveCreds);

    return new Promise((resolve, reject) => {
        sock.ev.on('connection.update', async (update) => {
            const { connection, lastDisconnect, qr } = update;

            if (qr) {
                console.log("Please scan the QR code to link WhatsApp.");
                qrcode.generate(qr, { small: true });
            }

            if (connection === 'close') {
                const shouldReconnect = lastDisconnect.error?.output?.statusCode !== 401;
                if (shouldReconnect) {
                    console.log('Connection closed, reconnecting...');
                    sendWhatsAppMessage(phone, message).then(resolve).catch(reject);
                } else {
                    console.error('Connection closed. Logged out.');
                    fs.rmSync(authDir, { recursive: true, force: true });
                    reject(new Error("Logged out"));
                }
            } else if (connection === 'open') {
                console.log('Connected to WhatsApp!');
                try {
                    // format phone number for WhatsApp
                    // Assuming phone is in format +91XXXXXXXXXX
                    const formattedPhone = phone.replace('+', '') + '@s.whatsapp.net';
                    await sock.sendMessage(formattedPhone, { text: message });
                    console.log('Message sent successfully!');
                    
                    // Close the socket gracefully after sending
                    setTimeout(() => {
                        sock.logout();
                        resolve();
                    }, 1000);
                    
                } catch (err) {
                    console.error('Failed to send message:', err);
                    reject(err);
                }
            }
        });
    });
}

// Read arguments
const args = process.argv.slice(2);
if (args.length < 2) {
    console.error("Usage: node send.js <phone> <message>");
    process.exit(1);
}

const phone = args[0];
const message = args.slice(1).join(" ");

sendWhatsAppMessage(phone, message)
    .then(() => process.exit(0))
    .catch((err) => process.exit(1));
