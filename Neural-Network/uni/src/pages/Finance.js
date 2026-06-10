import React from "react";
import { 
    Box, 
    Typography, 
    Paper, 
    Tabs, 
    Tab, 
    Table, 
    TableBody, 
    TableCell, 
    TableContainer, 
    TableHead, 
    TableRow,
    Chip
  } from '@mui/material';
  import { 
    Receipt as ReceiptIcon,
    Payment as PaymentIcon,
    AccountBalance as AccountBalanceIcon
  } from '@mui/icons-material';

function TabPanel(props) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

const transactions = [
    { id: 1, date: '2023-05-15', description: 'Tuition Fee', amount: 2500, status: 'paid', type: 'income' },
    { id: 2, date: '2023-05-10', description: 'Library Fine', amount: 25, status: 'unpaid', type: 'expense' },
    { id: 3, date: '2023-05-05', description: 'Dormitory Fee', amount: 1200, status: 'paid', type: 'income' },
];

export default function Finance() {
    const [value, setValue] = React.useState(0);

    const handleChange = (event, newValue) => {
        setValue(newValue);
    };

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
                Finance
            </Typography>

            <Paper sx={{ mb: 3 }}>
                <Tabs value={value} onChange={handleChange} aria-label='finance tabs'>
                    <Tab label="Transactions" icon={<ReceiptIcon />} iconPosition="start" />
                    <Tab label="Tuition Fees" icon={<PaymentIcon />} iconPosition="start" />
                    <Tab label="Budget" icon={<AccountBalanceIcon />} iconPosition="start" /> 
                </Tabs>
            </Paper>

            <TabPanel value={value} index={0}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Date</TableCell>
                                <TableCell>Description</TableCell>
                                <TableCell>Amount</TableCell>
                                <TableCell>Status</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {transactions.map((tx) => (
                                <TableRow key={tx.id}>
                                    <TableCell>{tx.date}</TableCell>
                                    <TableCell>{tx.description}</TableCell>
                                    <TableCell sx={{ color: tx.type === 'income' ? 'success.main' : 'error.main' }}>
                                        {tx.type === 'income' ? '+' : '-'}${tx.amount}
                                    </TableCell>
                                    <TableCell>
                                        <Chip 
                                        label={tx.status} 
                                        color={tx.status === 'paid' ? 'success' : 'error'}
                                        size="small"
                                        />
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </TabPanel>

            <TabPanel value={value} index={1}>
                <Typography>Tuition fee management content goes here</Typography>
            </TabPanel>

            <TabPanel value={value} index={2}>
                <Typography>Budget allocation content goes here</Typography>
            </TabPanel>
        </Box>
    );
}