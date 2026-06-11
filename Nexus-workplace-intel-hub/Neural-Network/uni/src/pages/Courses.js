import React, { useState } from 'react';
import { 
    Box, 
    Typography, 
    Paper, 
    Table, 
    TableBody, 
    TableCell, 
    TableContainer, 
    TableHead, 
    TableRow, 
    Button,
    TextField,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions
} from '@mui/material';
import { Add as AddIcon, Search as SearchIcon } from '@mui/icons-material';

const courseData = [
    { id: 1, code: 'CS101', name: 'Introduction to Programming', credits: 3, department: 'Computer Science' },
    { id: 2, code: 'MATH201', name: 'Calculus II', credits: 4, department: 'Mathematics' },
    { id: 3, code: 'ENG105', name: 'Academic Writing', credits: 3, department: 'English' },
    { id: 4, code: 'PHYS202', name: 'Modern Physics', credits: 4, department: 'Physics' },
];

export default function Courses() {
    const [open, setOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');

    const filteredCourses = courseData.filter(course => 
        course.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        course.code.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant='h4' gutterBottom>
                Courses
            </Typography>

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                <TextField
                    variant='outlined'
                    placeholder='Search courses...'
                    size='small'
                    InputProps={{
                        startAdornment: <SearchIcon color='action' sx={{ mr: 1 }} />
                    }}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
                <Button 
                    variant='contained'
                    startIcon={<AddIcon />}
                    onClick={() => setOpen(true)}
                >
                    Add Course
                </Button>
            </Box>

            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Course Code</TableCell>
                            <TableCell>Course Name</TableCell>
                            <TableCell>Credits</TableCell>
                            <TableCell>Department</TableCell>
                            <TableCell align='right'>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {filteredCourses.map((course) => (
                            <TableRow key={course.id}>
                                <TableCell>{course.code}</TableCell>
                                <TableCell>{course.name}</TableCell>
                                <TableCell>{course.credits}</TableCell>
                                <TableCell>{course.department}</TableCell>
                                <TableCell align="right">
                                    <Button size="small" color="primary">Edit</Button>
                                    <Button size="small" color="error">Delete</Button>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>

            <Dialog open={open} onClose={() => setOpen(false)}>
                <DialogTitle>Add New Course</DialogTitle>
                <DialogContent>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
                        <TextField label="Course Code" fullWidth />
                        <TextField label="Course Name" fullWidth />
                        <TextField label="Credits" type="number" fullWidth />
                        <TextField label="Department" fullWidth />
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpen(false)}>Cancel</Button>
                    <Button variant="contained">Save</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}